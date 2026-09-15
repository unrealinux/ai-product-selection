"""成本对齐：把供货价（1688 / 表格导入）配到零售商品（淘宝）上，算出真实毛利率。

为什么需要这一步
----------------
淘宝 A2A 只给零售价、没有成本，导入后所有商品毛利率都是 100%，
而毛利率是权重最高的维度 —— 它实际已饱和、不参与区分。
1688 表格导入则相反：有采购价、没有零售价。
把两边配起来，毛利率才第一次变成**真实数据**。

难点是匹配
----------
跨平台没有共同的商品 ID，只能靠标题。这里用的是：

1. **去掉营销噪音词**（新款/包邮/正品/爆款…）。这步很关键 ——
   这类词在两边标题里都大量出现，不去掉会凭空制造出大量假相似。
2. **字符二元组 Dice 系数**。中文不需要分词，二元组对标题这种短文本足够稳。
3. **规格 token 加成**（``304`` / ``500ml`` / ``316L``）。
   两边都有规格时按 0.65·Dice + 0.35·规格 Jaccard 混合 ——
   规格对不上（304 vs 316）应当显著扣分。

.. warning::

   标题匹配是**启发式**，不是精确配对。高分不代表同一款货，只代表"很可能是同类"。
   所以本模块默认只预览不写入（``dry_run``），并且**不覆盖已有成本**（除非 ``overwrite``），
   每条匹配都会带上相似度供人工复核。
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .enrich import PROVENANCE_SEP
from .models import ProductIn
from .sources.taobao import strip_highlight

logger = logging.getLogger(__name__)

#: 成本来源标注的分隔符。用它做幂等：重复对齐不会叠加多行
COST_SEP = "｜成本来源："

#: 营销噪音词。两边标题都大量出现，不去掉会制造假相似
NOISE_WORDS: tuple[str, ...] = (
    "包邮", "正品", "新款", "特价", "促销", "批发", "厂家直销", "一件代发",
    "支持定制", "官方旗舰店", "旗舰店", "专柜", "现货", "速发", "爆款", "网红",
    "同款", "自营", "限时", "清仓", "亏本", "冲量", "直销", "源头工厂", "工厂直供",
    "淘宝", "天猫", "1688", "阿里巴巴", "拼多多", "抖音",
)

_PUNCT = re.compile(r"[\s\-_/\\|,，。.、;；:：!！?？()（）\[\]【】{}<>《》\"'`~@#$%^&*+=]+")
_SPEC_TOKEN = re.compile(r"[a-z]*\d+(?:\.\d+)?[a-z]*")

#: 默认相似度阈值。低于它不自动应用
DEFAULT_THRESHOLD = 0.50
#: 低于阈值但这个倍数以上，算「低置信候选」，只在报告里列出
LOW_CONFIDENCE_RATIO = 0.6


# --------------------------------------------------------------------------- #
# 文本相似度
# --------------------------------------------------------------------------- #

def normalize_title(text: Any) -> str:
    """归一化标题：去高亮标签、转小写、去标点、去营销噪音词。"""
    value = strip_highlight(text).lower()
    value = _PUNCT.sub("", value)
    for word in NOISE_WORDS:
        value = value.replace(word.lower(), "")
    return value


def bigrams(text: Any) -> set[str]:
    """字符二元组集合（会先归一化）。中文无需分词，二元组对短文本已经够稳。"""
    value = normalize_title(text)
    if len(value) < 2:
        return {value} if value else set()
    return {value[i:i + 2] for i in range(len(value) - 1)}


def spec_tokens(text: Any) -> set[str]:
    """规格 token：``304`` / ``500ml`` / ``316l`` 这类带数字的片段。"""
    return set(_SPEC_TOKEN.findall(normalize_title(text)))


def dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_similarity(title_a: Any, title_b: Any) -> float:
    """标题相似度 0-1。

    两边都有规格 token 时，按 ``0.65·Dice + 0.35·规格Jaccard`` 混合 ——
    这样 ``304`` 与 ``316`` 这种关键差异会被显著扣分，而不是淹没在长标题里。
    """
    base = dice(bigrams(title_a), bigrams(title_b))

    specs_a, specs_b = spec_tokens(title_a), spec_tokens(title_b)
    if specs_a and specs_b:
        return round(0.65 * base + 0.35 * jaccard(specs_a, specs_b), 4)
    return round(base, 4)


def shared_specs(title_a: Any, title_b: Any) -> list[str]:
    return sorted(spec_tokens(title_a) & spec_tokens(title_b))


# --------------------------------------------------------------------------- #
# 匹配
# --------------------------------------------------------------------------- #

@dataclass
class CostMatch:
    """一条成本对齐结果。"""

    target: ProductIn
    supply: Optional[ProductIn]
    score: float
    cost: float
    method: str = "title"
    confidence: str = "high"
    shared_specs: list[str] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """用对齐后的成本算毛利率。"""
        if not self.target.price or self.target.price <= 0 or self.cost <= 0:
            return 0.0
        return round((self.target.price - self.cost) / self.target.price, 4)

    @property
    def supply_title(self) -> str:
        return self.supply.title if self.supply else ""

    def describe(self) -> str:
        if self.supply is None:
            return f"{self.target.title[:24]} ← 无匹配"
        return (f"{self.target.title[:22]} ← {self.supply.title[:22]}"
                f"（相似度 {self.score:.3f}，成本 {self.cost:.2f}）")


@dataclass
class LinkResult:
    """成本对齐的整体结果。"""

    matches: list[CostMatch] = field(default_factory=list)
    applied: int = 0
    skipped_existing: int = 0
    below_threshold: int = 0
    no_supply: int = 0
    dry_run: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[CostMatch]:
        return [m for m in self.matches if m.confidence == "high" and m.supply is not None]

    @property
    def low_confidence(self) -> list[CostMatch]:
        return [m for m in self.matches if m.confidence == "low" and m.supply is not None]

    @property
    def margins(self) -> list[float]:
        return [m.margin for m in self.accepted if m.margin > 0]

    def summary(self) -> str:
        parts = [
            f"待补成本 {len(self.matches)} 个",
            f"高置信匹配 {len(self.accepted)} 个",
            f"低置信候选 {len(self.low_confidence)} 个",
            f"无匹配 {self.no_supply} 个",
        ]
        if self.skipped_existing:
            parts.append(f"已有成本跳过 {self.skipped_existing} 个")
        if self.dry_run:
            parts.append("**预览模式，未写库**")
        else:
            parts.append(f"已写入 {self.applied} 个")
        text = "；".join(parts) + "。"
        if self.margins:
            text += (
                f" 对齐后毛利率：中位数 {statistics.median(self.margins):.1%}，"
                f"区间 {min(self.margins):.1%} ~ {max(self.margins):.1%}。"
            )
        return text


def pick_supply(
    target: ProductIn,
    supplies: Sequence[ProductIn],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_shared_specs: int = 0,
) -> tuple[Optional[ProductIn], float]:
    """在供货池里为一条商品找最相似的来源。

    Returns:
        ``(最佳来源 或 None, 相似度)``
    """
    best: Optional[ProductIn] = None
    best_score = 0.0
    for supply in supplies:
        if supply.cost <= 0:
            continue
        score = title_similarity(target.title, supply.title)
        if score <= best_score:
            continue
        if min_shared_specs and len(shared_specs(target.title, supply.title)) < min_shared_specs:
            continue
        best, best_score = supply, score
    if best_score < threshold * LOW_CONFIDENCE_RATIO:
        return None, best_score
    return best, best_score


def link_costs(
    products: Sequence[ProductIn],
    *,
    target_source: str = "",
    supply_source: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    overwrite: bool = False,
    min_shared_specs: int = 0,
    dry_run: bool = True,
    limit: int = 0,
) -> LinkResult:
    """把有成本的商品的采购价，对齐到缺成本的商品上。

    Args:
        target_source: 只补这个来源的商品；留空表示「所有缺成本的商品」。
        supply_source: 只从这个来源找成本；留空表示「所有有成本的商品」。
        threshold: 相似度阈值，低于它只作为低置信候选列出、不应用。
        overwrite: 是否覆盖已有成本。默认 False —— 已有的成本更可信。
        min_shared_specs: 要求至少共享几个规格 token（如 304/500ml），0 表示不要求。
        dry_run: True 时只算不写（本函数本身不写库，由调用方决定）。
        limit: 最多处理多少个待补商品，0 表示不限。
    """
    result = LinkResult(dry_run=dry_run)

    targets = [p for p in products if p.cost <= 0
               and (not target_source or p.source == target_source)]
    if not targets:
        result.warnings.append(
            "没有找到「缺成本」的商品。"
            + (f"（限定了来源 {target_source!r}）" if target_source else "")
        )
        return result

    # 有成本的商品不可能是待补目标，因此不需要再排除 target_source
    supplies = [p for p in products if p.cost > 0
                and (not supply_source or p.source == supply_source)]
    if not supplies:
        result.warnings.append(
            "没有找到「有成本」的商品作为供货来源。"
            "请先用表格导入 1688 商品表，或在商品录入页填成本。"
        )
        return result

    if limit:
        targets = targets[:limit]

    for target in targets:
        supply, score = pick_supply(
            target, supplies, threshold=threshold, min_shared_specs=min_shared_specs
        )
        if supply is None:
            result.no_supply += 1
            result.matches.append(CostMatch(target=target, supply=None, score=score,
                                            cost=0.0, method="none", confidence="none"))
            continue

        confidence = "high" if score >= threshold else "low"
        if confidence == "low":
            result.below_threshold += 1
        result.matches.append(CostMatch(
            target=target, supply=supply, score=score, cost=supply.cost,
            method="title", confidence=confidence,
            shared_specs=shared_specs(target.title, supply.title),
        ))

    # 批发价高于零售价不符合常识 —— 这通常不是「亏本货」而是「匹配错了」，
    # 必须主动说出来，否则用户会拿一个错误结论去做决策。
    negative = [m for m in result.accepted if m.margin <= 0]
    if negative:
        result.warnings.append(
            f"{len(negative)} 条匹配的采购价 ≥ 零售价，出现负毛利。"
            "批发价高于零售价不符合常识，**很可能是匹配错了**"
            "（标题相似不等于同一款货），请逐条核对后再决定是否采用。"
        )

    return result


# --------------------------------------------------------------------------- #
# 写回
# --------------------------------------------------------------------------- #

def cost_note(note: str, match: CostMatch) -> str:
    """把成本来源写进 note（幂等：重复对齐不会叠加）。"""
    body = (note or "").split(COST_SEP)[0]
    if match.supply is None:
        return body
    label = "低置信" if match.confidence == "low" else "相似度"
    detail = (f"{COST_SEP}采购价 {match.cost:.2f} 元来自「{match.supply.title[:28]}」"
              f"（{label} {match.score:.3f}"
              + (f"，共同规格 {'/'.join(match.shared_specs)}" if match.shared_specs else "")
              + "），需人工复核")
    return f"{body}{detail}"


def with_cost(match: CostMatch) -> ProductIn:
    """产出带新成本与新 note 的商品对象（不落库）。"""
    return match.target.model_copy(update={
        "cost": round(match.cost, 2),
        "note": cost_note(match.target.note, match),
    })


def apply_matches(result: LinkResult, *, include_low_confidence: bool = False,
                  db_path: Any = None) -> int:
    """把匹配结果写回数据库，返回写入条数。

    ``note`` 里会保留成本来源与相似度，便于事后复核。
    """
    from . import db

    written = 0
    for match in result.matches:
        if match.supply is None:
            continue
        if match.confidence == "low" and not include_low_confidence:
            continue
        db.upsert_product(with_cost(match), db_path)
        written += 1
    result.applied = written
    result.dry_run = False
    return written


def margin_report(products: Iterable[ProductIn]) -> dict[str, Any]:
    """统计一批商品的毛利率分布，用于判断成本是否对齐成功。"""
    usable = [p for p in products if p.price > 0 and p.cost > 0]
    if not usable:
        return {"count": 0, "median": 0.0, "min": 0.0, "max": 0.0,
                "saturated": 0, "missing_cost": 0}
    margins = [(p.price - p.cost) / p.price for p in usable]
    return {
        "count": len(usable),
        "median": round(statistics.median(margins), 4),
        "min": round(min(margins), 4),
        "max": round(max(margins), 4),
        # 毛利率 ≥ 95% 基本等于没成本，是数据质量问题
        "saturated": sum(1 for value in margins if value >= 0.95),
        "missing_cost": sum(1 for p in products if p.cost <= 0),
    }
