"""规则打分引擎。

设计原则：
1. 纯函数、无副作用，方便单测与复用；
2. 每个维度归一化到 0-100，再按权重加权求总分；
3. 大模型只做「点评 + 有限修正」，不推翻规则结果，避免不可解释。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DEFAULT_WEIGHTS

#: 物流友好度的重量区间（kg）：<=1kg 满分，>=5kg 零分
SHIPPING_BEST_KG = 1.0
SHIPPING_WORST_KG = 5.0

#: 毛利率区间：<=0 零分，>=60% 满分
MARGIN_BEST = 0.60

#: 大模型允许的最大修正幅度（分）
MAX_LLM_ADJUSTMENT = 10.0


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """把数值截断到 [low, high]。"""
    return max(low, min(high, value))


def linear_map(value: float, in_low: float, in_high: float,
               out_low: float = 0.0, out_high: float = 100.0) -> float:
    """把 value 从 [in_low, in_high] 线性映射到 [out_low, out_high] 并截断。"""
    if in_high == in_low:
        return out_low
    ratio = (value - in_low) / (in_high - in_low)
    return clamp(out_low + ratio * (out_high - out_low), min(out_low, out_high), max(out_low, out_high))


def log_scale(value: float, best: float, low: float = 0.0, high: float = 100.0) -> float:
    """对数映射：value 达到 ``best`` 时得 ``high`` 分，0 或负数得 ``low`` 分。

    用于销量、在售商品数这类**跨数量级**的指标 —— 销量 100 和 10000 的差距，
    远大于 10000 和 19900 的差距，线性映射会失真。
    """
    if value <= 0 or best <= 0:
        return low
    ratio = math.log10(value + 1) / math.log10(best + 1)
    return round(clamp(low + ratio * (high - low), low, high), 2)


def profit_margin(price: float, cost: float) -> float:
    """毛利率 = (售价 - 成本) / 售价，售价为 0 时返回 0。"""
    if price <= 0:
        return 0.0
    return (price - cost) / price


def score_dimensions(product) -> dict[str, float]:
    """计算各维度得分（均为 0-100，越大越好）。"""
    margin = profit_margin(product.price, product.cost)
    return {
        "demand": clamp(product.heat),
        "competition": clamp(100.0 - product.competition),
        "margin": linear_map(margin, 0.0, MARGIN_BEST),
        "shipping": linear_map(product.weight_kg, SHIPPING_BEST_KG, SHIPPING_WORST_KG, 100.0, 0.0),
        "repurchase": clamp(product.repurchase),
        "compliance": clamp(100.0 - product.compliance_risk),
        "virality": clamp(product.virality),
    }


def weighted_total(dimensions: dict[str, float],
                   weights: dict[str, float] | None = None) -> float:
    """按权重求加权总分，权重和不为 1 时自动归一化。"""
    weights = weights or DEFAULT_WEIGHTS
    total_weight = sum(weights.get(key, 0.0) for key in dimensions)
    if total_weight <= 0:
        return 0.0
    score = sum(dimensions.get(key, 0.0) * weights.get(key, 0.0) for key in dimensions)
    return round(score / total_weight, 2)


def grade_of(total: float) -> str:
    """总分转等级。"""
    if total >= 85:
        return "S"
    if total >= 75:
        return "A"
    if total >= 65:
        return "B"
    if total >= 55:
        return "C"
    return "D"


def build_advice(product, dimensions: dict[str, float], total: float) -> str:
    """根据短板维度生成可执行的选品建议。"""
    margin = profit_margin(product.price, product.cost)
    tips: list[str] = []

    if margin <= 0:
        tips.append("当前售价未覆盖成本，先核算定价或更换供应链")
    elif margin < 0.25:
        tips.append(f"毛利率仅 {margin:.0%}，低于健康线 25%，建议提价或压成本")

    if product.competition >= 70:
        tips.append(f"竞争度 {product.competition:.0f} 偏高，需要差异化卖点或细分人群切入")
    if product.heat < 40:
        tips.append("需求热度偏低，建议先用小预算测款验证真实需求")
    if product.weight_kg > 3:
        tips.append(f"单件 {product.weight_kg:g}kg，物流成本占比高，考虑轻量化或自发货改为云仓")
    if product.compliance_risk >= 50:
        tips.append("合规风险偏高，上架前确认资质、认证与广告法表述")
    if product.repurchase >= 70:
        tips.append("复购潜力不错，可设计耗材/组合装提升 LTV")
    if product.virality >= 70:
        tips.append("内容传播力强，适合短视频种草 + 达人分销放大")

    if not tips:
        tips.append("各项指标均衡，可直接进入小批量测款阶段")

    level = grade_of(total)
    head = {
        "S": "强烈推荐主推",
        "A": "推荐上架",
        "B": "可测款观察",
        "C": "谨慎尝试",
        "D": "建议放弃",
    }[level]
    return f"【{head}】" + "；".join(tips) + "。"


@dataclass
class RuleResult:
    """规则打分结果。"""

    total: float
    grade: str
    dimensions: dict[str, float]
    profit_margin: float
    advice: str


def rule_score(product, weights: dict[str, float] | None = None) -> RuleResult:
    """对单个商品执行完整规则打分。"""
    dimensions = score_dimensions(product)
    total = weighted_total(dimensions, weights)
    return RuleResult(
        total=total,
        grade=grade_of(total),
        dimensions=dimensions,
        profit_margin=round(profit_margin(product.price, product.cost), 4),
        advice=build_advice(product, dimensions, total),
    )
