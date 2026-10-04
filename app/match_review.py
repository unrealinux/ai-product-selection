"""用大模型复核「低置信」的标题匹配。

为什么需要它
------------
``costlink.py`` 的标题相似度是启发式的，分数落在阈值下方一档的候选最尴尬：
既不敢自动应用，也不该直接丢掉。而这种「灰区」恰恰最需要判断力 ——
Dice 系数看不出：

- ``304`` vs ``316L`` 是不同材质（规格 token 只能覆盖到一部分写法）
- 单只 vs ``4 件装`` 是不同商品（数量差异不体现在相似度上）
- ``降噪耳机头戴式`` vs ``降噪耳机入耳式`` 用途相同但形态不同

大模型能读出这些语义差异，所以让它只对灰区候选做一次「是不是同一款货」的判断。

设计原则
--------
1. **只在灰区花钱**：只复核 ``confidence == "low"`` 的候选，不动高置信匹配。
2. **宁可不判**：模型自评置信度低于阈值时**保持低置信**，既不升级也不删除。
3. **可审计**：``note`` 写成「低置信经大模型复核通过（置信 85）」，不伪装成硬数据。
4. **缓存**：按 ``(零售标题, 供货标题)`` 归一化后缓存，重复运行不重复花 token。
5. **失败不阻塞**：未配置 LLM 或调用失败时原样返回，成本对齐流程继续。

.. warning::
   大模型复核会**降低但不能消除**错配风险。复核通过的匹配仍然带「需人工复核」标注。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import llm
from .config import settings
from .costlink import CostMatch, LinkResult, normalize_title

logger = logging.getLogger(__name__)

#: 单次请求放几组候选（组数太多会稀释模型注意力）
DEFAULT_BATCH_SIZE = 8

SYSTEM_PROMPT = """你是电商供应链选品专家。用户会给出若干组「零售商品标题」与「候选供货商品标题」，
以及它们的标题相似度。请判断每一组**是否为同一款商品**（而不是仅仅同类）。

判断要点：
- 规格必须一致：容量 / 尺寸 / 材质 / 型号不同就是不同商品（304 vs 316L、500ml vs 150ml）。
- 数量或套装数不同（单只 vs 4 件装）算不同商品。
- 用途相同但形态不同（入耳式 vs 头戴式、迷你 vs 大容量）不算同一款。
- 标题措辞不同、但规格与形态指向同一款货，算同一款。
- 批发价高于零售价通常意味着匹配错了。

只输出 JSON（不要包含 markdown 代码块），格式：
{"items": [{"index": 0, "same_product": true, "confidence": 0-100, "reason": "15 字以内中文理由"}]}

confidence 是你对自己判断的把握，不是标题相似度。拿不准就填低分。"""


# --------------------------------------------------------------------------- #
# 结果结构
# --------------------------------------------------------------------------- #

@dataclass
class ReviewVerdict:
    """模型对一组候选的判断。"""

    index: int
    same_product: Optional[bool] = None
    confidence: int = 0
    reason: str = ""


@dataclass
class ReviewReport:
    """一次复核的整体情况。"""

    reviewed: int = 0
    promoted: int = 0
    rejected: int = 0
    kept: int = 0
    cached: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not (self.reviewed or self.cached):
            return "未复核任何候选。"
        parts = [f"复核 {self.reviewed} 组"]
        if self.cached:
            parts.append(f"命中缓存 {self.cached} 组")
        parts.append(f"升级为高置信 {self.promoted} 组")
        parts.append(f"判定非同款 {self.rejected} 组")
        parts.append(f"仍不确定 {self.kept} 组")
        return "；".join(parts) + "。"


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #

class ReviewCache:
    """按 ``(零售标题, 供货标题)`` 归一化指纹缓存的 JSON 文件。"""

    def __init__(self, path: Path | str | None = None, *, enabled: bool = True) -> None:
        self.path = Path(path) if path else settings.match_review_cache_path
        self.enabled = enabled
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if self.enabled:
            self._load()

    @staticmethod
    def make_key(target_title: str, supply_title: str) -> str:
        raw = f"{normalize_title(target_title)}|{normalize_title(supply_title)}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._data = {
                    str(key): value for key, value in payload.items()
                    if isinstance(value, dict)
                }
        except Exception as exc:  # noqa: BLE001 - 缓存损坏不应影响主流程
            logger.warning("复核缓存读取失败，将忽略：%s", exc)
            self._data = {}

    def get(self, target_title: str, supply_title: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        return self._data.get(self.make_key(target_title, supply_title))

    def put(self, target_title: str, supply_title: str, verdict: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self._data[self.make_key(target_title, supply_title)] = dict(verdict)
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._dirty = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("复核缓存写入失败：%s", exc)

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# 提示词与解析
# --------------------------------------------------------------------------- #

def build_pair_prompt(pairs: Sequence[tuple[int, CostMatch]]) -> str:
    """构造用户提示词。``pairs`` 里的 index 会原样回传，用于对齐结果。"""
    blocks: list[str] = []
    for index, match in pairs:
        specs = "/".join(match.shared_specs) or "无"
        price = f"{match.target.price:.2f} 元" if match.target.price else "未知"
        cost = f"{match.cost:.2f} 元" if match.cost else "未知"
        blocks.append(
            f"第 {index} 组\n"
            f"  零售商品：{match.target.title}\n"
            f"  候选供货：{match.supply_title or '（无）'}\n"
            f"  标题相似度：{match.score:.3f}\n"
            f"  共同规格：{specs}\n"
            f"  零售价：{price}　批发价：{cost}"
        )
    return "\n\n".join(blocks) + "\n\n请逐一判断，按 index 输出 JSON。"


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "是", "同款"}:
        return True
    if text in {"false", "no", "n", "0", "否", "非同款"}:
        return False
    return None


def parse_verdicts(payload: Any, size: int,
                   allowed: Optional[set[int]] = None) -> dict[int, ReviewVerdict]:
    """解析模型返回，返回 ``{index: ReviewVerdict}``。

    Args:
        size: 本批候选数量，超出范围的 index 一律丢弃（防止模型编号串位）。
        allowed: 允许出现的 index 集合；为 ``None`` 表示 ``0 .. size-1``。
    """
    if isinstance(payload, dict):
        raw_items = payload.get("items")
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = None
    if not isinstance(raw_items, list):
        return {}

    valid = allowed if allowed is not None else set(range(size))
    parsed: dict[int, ReviewVerdict] = {}
    for position, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            index = position
        if index not in valid:
            continue
        try:
            confidence = int(float(item.get("confidence", 0) or 0))
        except (TypeError, ValueError):
            confidence = 0
        parsed[index] = ReviewVerdict(
            index=index,
            same_product=_as_bool(item.get("same_product", item.get("same"))),
            confidence=max(0, min(100, confidence)),
            reason=str(item.get("reason") or "").strip()[:60],
        )
    return parsed


# --------------------------------------------------------------------------- #
# 应用判断
# --------------------------------------------------------------------------- #

def apply_verdicts(matches: Sequence[CostMatch], verdicts: Mapping[int, ReviewVerdict],
                   *, min_confidence: int) -> tuple[int, int, int]:
    """把判断写回匹配对象，返回 ``(升级, 判非同款, 仍不确定)``。

    ``index`` 是候选在 ``matches`` 里的下标。置信度不足时**保持原样** ——
    宁可不判，也不把不确定的判断当成结论。
    """
    promoted = rejected = kept = 0
    for index, verdict in verdicts.items():
        if not 0 <= index < len(matches):
            continue
        match = matches[index]
        if match.supply is None:
            continue
        match.review_reason = verdict.reason
        match.review_confidence = verdict.confidence

        if verdict.same_product is None or verdict.confidence < min_confidence:
            kept += 1
            continue
        if verdict.same_product:
            match.confidence = "high"
            match.method = "llm_review"
            promoted += 1
        else:
            match.confidence = "rejected"
            rejected += 1
    return promoted, rejected, kept


def review_matches(
    result: LinkResult,
    *,
    min_confidence: Optional[int] = None,
    batch_size: Optional[int] = None,
    cache: Optional[ReviewCache] = None,
    report: Optional[ReviewReport] = None,
) -> ReviewReport:
    """复核 ``result`` 里的低置信候选，就地修改它们的 ``confidence``。

    Args:
        min_confidence: 模型自评置信度门槛，低于它保持低置信。默认取配置值。
        batch_size: 每批多少组。
        cache: 复核缓存；传 ``ReviewCache(enabled=False)`` 可关闭。
        report: 复用已有报告对象。

    Returns:
        :class:`ReviewReport`。未配置 LLM 或没有候选时 ``reviewed`` 为 0，不抛异常。
    """
    report = report if report is not None else ReviewReport()
    min_confidence = (
        settings.match_review_min_confidence if min_confidence is None else min_confidence
    )
    batch_size = max(1, int(batch_size or settings.match_review_batch_size))

    candidates = result.low_confidence
    if not candidates:
        report.notes.append("没有需要复核的低置信候选。")
        return report
    if not llm.is_available():
        report.notes.append(
            "未配置 APS_LLM_*，跳过复核。低置信候选保持原样，不会因为没有复核被丢弃。"
        )
        return report

    cache = cache if cache is not None else ReviewCache()
    verdicts: dict[int, ReviewVerdict] = {}
    pending: list[int] = []

    for index, match in enumerate(candidates):
        cached = cache.get(match.target.title, match.supply_title)
        if cached is None:
            pending.append(index)
            continue
        report.cached += 1
        verdicts[index] = ReviewVerdict(
            index=index,
            same_product=_as_bool(cached.get("same_product")),
            confidence=int(cached.get("confidence") or 0),
            reason=str(cached.get("reason") or ""),
        )

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        pairs = [(index, candidates[index]) for index in chunk]
        content = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_pair_prompt(pairs)},
            ],
            temperature=0.1,
        )
        report.reviewed += len(chunk)
        if not content:
            report.errors.append(
                f"大模型无返回（第 {start // batch_size + 1} 批，{len(chunk)} 组）"
            )
            continue

        parsed = parse_verdicts(llm.extract_json(content), len(chunk), allowed=set(chunk))
        if not parsed:
            report.errors.append(
                f"大模型返回无法解析（第 {start // batch_size + 1} 批，{len(chunk)} 组）"
            )
            continue

        for index, verdict in parsed.items():
            verdicts[index] = verdict
            match = candidates[index]
            cache.put(match.target.title, match.supply_title, {
                "same_product": verdict.same_product,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
            })

    cache.save()
    promoted, rejected, kept = apply_verdicts(candidates, verdicts,
                                              min_confidence=min_confidence)
    report.promoted, report.rejected, report.kept = promoted, rejected, kept
    if report.rejected:
        report.notes.append(
            "被判定为「非同款」的候选不会写入。它们保留在报告里，便于核对模型的判断。"
        )
    return report
