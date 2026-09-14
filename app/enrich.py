"""补齐精选联盟接口未提供的维度：重量、复购潜力、合规风险。

三条来源，优先级从高到低：

1. **商品详情接口** ``product.detail`` —— 仅对**自己店铺**的商品有效。
   官方对该接口 ``product_id`` 的说明是「抖店系统生成，**店铺下唯一**」，
   因此精选联盟里其他商家的商品查不到（会返回「商品不存在」）。
2. **大模型估算** —— 任意商品，按标题 + 类目推断。结果一律标注为估算值。
3. **保持缺省值** —— 并继续在 ``note`` 里标注「需人工复核」。

设计原则：**绝不把估算值伪装成接口数据**。每个被补齐的字段都会写进
``note`` 的数据来源说明，调用方也能从 :class:`EnrichReport` 拿到明细。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import llm
from .config import settings
from .models import ProductIn
from .sources.douyin import DouyinAPIError, DouyinClient, parse_weight_kg

logger = logging.getLogger(__name__)

#: note 里数据来源说明的分隔符（与 app.sources.douyin.to_product 保持一致）
PROVENANCE_SEP = "｜数据说明："

#: 需要补齐的字段 → 中文名
TARGET_FIELDS: dict[str, str] = {
    "weight_kg": "重量",
    "repurchase": "复购",
    "compliance_risk": "合规",
}

FIELD_LIMITS: dict[str, tuple[float, float]] = {
    "weight_kg": (0.01, 50.0),
    "repurchase": (0.0, 100.0),
    "compliance_risk": (0.0, 100.0),
}

LLM_SYSTEM_PROMPT = """你是跨境电商选品与供应链专家。用户会给出商品标题与类目，
请估算三个无法从接口获得的字段：

- weight_kg：单件商品（含常规包装）的运输重量，单位 kg，保留 2 位小数。
  参考量级：手机壳 0.05、T 恤 0.25、保温杯 0.35、咖啡粉 30 条 0.4、
  折叠椅 3.2、小型家电 1.5、智能垃圾桶 2.9。
- repurchase：复购潜力 0-100。食品/日化/耗材偏高（60-90）；
  家电/家具/耐用品偏低（5-25）。
- compliance_risk：合规风险 0-100。涉及医疗器械、儿童玩具、化妆品、
  保健或疗效宣称、锂电池运输的偏高（40-80）；普通日用百货偏低（5-20）。

只输出 JSON，不要 markdown 代码块，不要解释。格式：
{"items":[{"index":0,"weight_kg":0.35,"repurchase":25,"compliance_risk":15,
"reason":"保温杯属耐用品复购低，需食品接触材料报告"}]}

硬性要求：
1. index 必须与输入编号一致；
2. 三个字段都必须给出具体数值，不要返回 null；
3. reason 不超过 40 字，说明主要判断依据。"""


@dataclass
class EnrichReport:
    """补齐过程的明细，供 CLI / UI 展示。"""

    detail_attempted: int = 0
    detail_filled: int = 0
    detail_failed: int = 0
    llm_requested: int = 0
    llm_filled: int = 0
    llm_cached: int = 0
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def filled(self) -> int:
        """被补齐过至少一个字段的商品数量（粗略统计，按字段累加）。"""
        return self.detail_filled + self.llm_filled

    def summary(self) -> str:
        parts = []
        if self.detail_attempted:
            parts.append(
                f"详情接口：尝试 {self.detail_attempted} 个，成功 {self.detail_filled} 个，"
                f"失败 {self.detail_failed} 个"
            )
        if self.llm_requested:
            parts.append(
                f"大模型估算：请求 {self.llm_requested} 个，成功 {self.llm_filled} 个"
                + (f"（命中缓存 {self.llm_cached} 个）" if self.llm_cached else "")
            )
        if not parts:
            return "未执行任何补齐操作。"
        return "；".join(parts) + "。"


# --------------------------------------------------------------------------- #
# 大模型估算结果缓存（估算要花 token，同一商品不必重复问）
# --------------------------------------------------------------------------- #

class EstimateCache:
    """按 ``标题 + 类目`` 做键的 JSON 文件缓存。"""

    def __init__(self, path: Path | str | None = None, *, enabled: bool = True) -> None:
        self.path = Path(path) if path else settings.enrich_cache_path
        self.enabled = enabled
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if self.enabled:
            self._load()

    @staticmethod
    def make_key(product: ProductIn) -> str:
        raw = f"{product.title}|{product.category}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._data = {
                    str(key): value
                    for key, value in payload.items()
                    if isinstance(value, dict)
                }
        except Exception as exc:  # noqa: BLE001 - 缓存损坏不应影响主流程
            logger.warning("估算缓存读取失败，将忽略：%s", exc)
            self._data = {}

    def get(self, product: ProductIn) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        return self._data.get(self.make_key(product))

    def put(self, product: ProductIn, estimate: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self._data[self.make_key(product)] = dict(estimate)
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
            logger.warning("估算缓存写入失败：%s", exc)

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def _coerce(value: Any, low: float, high: float) -> Optional[float]:
    """把模型返回值转成受限浮点数；无效返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(max(low, min(high, number)), 4)


def note_body(note: str) -> str:
    """去掉 note 尾部的数据来源说明。"""
    return (note or "").split(PROVENANCE_SEP)[0]


def describe_sources(sources: Mapping[str, str]) -> str:
    """生成 note 尾部的数据来源说明。"""
    if not sources:
        return "热度/竞争度/毛利率/传播力由接口字段推导；重量、复购、合规为缺省值，需人工复核"
    bits = [f"{TARGET_FIELDS.get(k, k)}={v}" for k, v in sources.items()]
    return "热度/竞争度/毛利率由接口字段推导；" + "、".join(bits)


def with_sources(note: str, sources: Mapping[str, str]) -> str:
    """把数据来源说明写回 note。"""
    return f"{note_body(note)}{PROVENANCE_SEP}{describe_sources(sources)}"


# --------------------------------------------------------------------------- #
# 路径 1：商品详情接口（仅自己店铺的商品）
# --------------------------------------------------------------------------- #

def fill_weight_from_detail(
    products: Sequence[ProductIn],
    client: DouyinClient,
    *,
    limit: int = 0,
    report: Optional[EnrichReport] = None,
    sources: Optional[dict[int, dict[str, str]]] = None,
) -> list[ProductIn]:
    """用 ``product.detail`` 补齐重量。

    只对**已授权店铺自己的商品**有效。精选联盟商品的 ``product_id`` 属于其他商家，
    调用会返回「商品不存在」，这类失败会被计入 ``report.detail_failed`` 而不中断流程。

    Args:
        limit: 最多尝试多少个商品（每个商品 1 次接口调用）。``0`` 表示全部。
    """
    report = report if report is not None else EnrichReport()
    sources = sources if sources is not None else {}
    items = list(products)

    candidates = [i for i, item in enumerate(items) if _extract_product_id(item)]
    if limit:
        candidates = candidates[:limit]

    for index in candidates:
        item = items[index]
        product_id = _extract_product_id(item)
        if not product_id:
            continue
        report.detail_attempted += 1
        try:
            detail = client.get_product_detail(product_id=product_id)
        except DouyinAPIError as exc:
            report.detail_failed += 1
            if len(report.errors) < 5:
                report.errors.append(f"{item.title[:20]}：{exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            report.detail_failed += 1
            if len(report.errors) < 5:
                report.errors.append(f"{item.title[:20]}：{exc}")
            continue

        weight = parse_weight_kg(detail)
        if weight:
            items[index] = item.model_copy(update={"weight_kg": weight})
            sources.setdefault(index, {})["weight_kg"] = "商品详情接口"
            report.detail_filled += 1

    if report.detail_attempted and not report.detail_filled:
        report.notes.append(
            "商品详情接口未补到任何重量。该接口只能查已授权店铺自己的商品，"
            "精选联盟里其他商家的商品查不到（会返回「商品不存在」）。"
        )
    return items


def _extract_product_id(product: ProductIn) -> Optional[str]:
    """取来源平台的商品 ID。

    优先用显式字段 ``external_id``；旧数据（该字段引入前导入的）回退到解析 note，
    以便升级后无需重新导入。
    """
    if product.external_id:
        return product.external_id.strip() or None

    marker = "商品ID "
    note = product.note or ""
    if marker not in note:
        return None
    tail = note.split(marker, 1)[1]
    token = tail.split(" ", 1)[0].strip()
    return token or None


# --------------------------------------------------------------------------- #
# 路径 2：大模型估算
# --------------------------------------------------------------------------- #

def _build_user_prompt(batch: Sequence[ProductIn]) -> str:
    lines = []
    for index, item in enumerate(batch):
        lines.append(f'{index}. 标题：{item.title} ｜ 类目：{item.category}')
    return "\n".join(lines)


def _parse_estimates(payload: Any, size: int) -> dict[int, dict[str, float]]:
    """把模型返回解析为 {index: {field: value}}。"""
    items: Any = payload
    if isinstance(payload, Mapping):
        items = payload.get("items") or payload.get("data") or []
    if not isinstance(items, list):
        return {}

    parsed: dict[int, dict[str, float]] = {}
    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < size:
            continue
        values: dict[str, float] = {}
        for name, (low, high) in FIELD_LIMITS.items():
            number = _coerce(entry.get(name), low, high)
            if number is not None:
                values[name] = number
        if values.get("weight_kg") is not None:
            values.setdefault("weight_kg", 0.0)
        if values:
            parsed[index] = values
    return parsed


def estimate_with_llm(
    products: Sequence[ProductIn],
    *,
    batch_size: Optional[int] = None,
    cache: Optional[EstimateCache] = None,
    report: Optional[EnrichReport] = None,
    sources: Optional[dict[int, dict[str, str]]] = None,
) -> list[ProductIn]:
    """用大模型估算 ``weight_kg`` / ``repurchase`` / ``compliance_risk``。

    未配置 LLM 或调用失败时原样返回，不影响主流程。
    """
    report = report if report is not None else EnrichReport()
    sources = sources if sources is not None else {}
    items = list(products)

    if not items:
        return items
    if not llm.is_available():
        report.notes.append("未配置 APS_LLM_* ，跳过重量/复购/合规的估算，保持缺省值。")
        return items

    batch_size = batch_size or settings.enrich_batch_size
    batch_size = max(1, min(int(batch_size), 20))
    cache = cache if cache is not None else EstimateCache()

    pending: list[int] = []
    for index, item in enumerate(items):
        cached = cache.get(item)
        if cached:
            report.llm_cached += 1
            values = _parse_estimates({"items": [dict(cached, index=0)]}, 1).get(0, {})
            if values:
                items[index] = _apply(item, values)
                sources.setdefault(index, {})["weight_kg"] = "大模型估算"
                sources[index]["repurchase"] = "大模型估算"
                sources[index]["compliance_risk"] = "大模型估算"
                report.llm_filled += 1
            continue
        pending.append(index)

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        batch = [items[i] for i in chunk]
        content = llm.chat(
            [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(batch)},
            ],
            temperature=0.2,
        )
        report.llm_requested += len(chunk)
        if not content:
            report.errors.append(f"大模型无返回（第 {start // batch_size + 1} 批，{len(chunk)} 条）")
            continue

        parsed = _parse_estimates(llm.extract_json(content), len(chunk))
        if not parsed:
            report.errors.append(
                f"大模型返回无法解析（第 {start // batch_size + 1} 批）：{content[:120]}"
            )
            continue

        for offset, values in parsed.items():
            index = chunk[offset]
            items[index] = _apply(items[index], values)
            field_sources = sources.setdefault(index, {})
            for name in values:
                field_sources[name] = "大模型估算"
            report.llm_filled += 1
            cache.put(items[index], {"index": 0, **values})

    cache.save()
    if report.llm_filled:
        report.notes.append(
            "重量/复购/合规为**大模型估算值**，非接口数据，已写入 note 的数据来源说明。"
        )
    return items


def _apply(product: ProductIn, values: Mapping[str, float]) -> ProductIn:
    """把估算值写回商品（只覆盖确实返回了的字段）。"""
    updates = {name: value for name, value in values.items() if name in FIELD_LIMITS}
    return product.model_copy(update=updates) if updates else product


# --------------------------------------------------------------------------- #
# 编排入口
# --------------------------------------------------------------------------- #

def enrich(
    products: Iterable[ProductIn],
    *,
    client: Optional[DouyinClient] = None,
    use_llm: bool = True,
    detail_limit: int = 0,
    batch_size: Optional[int] = None,
    cache: Optional[EstimateCache] = None,
    use_cache: bool = True,
) -> tuple[list[ProductIn], EnrichReport]:
    """按优先级补齐重量 / 复购 / 合规，并重写 note 里的数据来源说明。

    Args:
        client: 传入 ``DouyinClient`` 才会尝试走商品详情接口（仅自己店铺商品有效）。
        use_llm: 是否用大模型估算。
        detail_limit: 商品详情接口最多尝试多少个商品；``0`` 表示全部。
        use_cache: 是否使用估算结果缓存。

    Returns:
        ``(补齐后的商品列表, 明细报告)``
    """
    report = EnrichReport()
    items = list(products)
    sources: dict[int, dict[str, str]] = {index: {} for index in range(len(items))}

    if client is not None:
        items = fill_weight_from_detail(
            items, client, limit=detail_limit, report=report, sources=sources
        )

    if use_llm:
        items = estimate_with_llm(
            items,
            batch_size=batch_size,
            cache=cache if cache is not None else EstimateCache(enabled=use_cache),
            report=report,
            sources=sources,
        )

    enriched = [
        item.model_copy(update={"note": with_sources(item.note, sources[index])})
        if sources.get(index)
        else item
        for index, item in enumerate(items)
    ]
    return enriched, report
