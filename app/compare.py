"""打分运行快照的对比与权重敏感度分析。

回答两个具体问题：

1. **换一套权重，结论会变吗？** —— :func:`compare_runs`
   两次运行都存了各自的总额分与各维度得分，因此可以逐商品算出排名变化、
   秩相关系数（Spearman ρ）、Top-N 重合度，以及变动最大的商品。

2. **哪个维度在真正决定排序？** —— :func:`weight_sensitivity`
   把某个维度的权重置 0 后重算排名，与原排名求相关。ρ 越低说明该维度影响力越大；
   若某维度置 0 后 ρ≈1，说明它几乎不参与决策 —— 那么它取接口值还是大模型估算
   也就无所谓了。这正是判断「用大模型覆盖热度值不值」的依据。

纯函数实现，不碰数据库，便于单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .config import DIMENSION_LABELS
from .scoring import weighted_total
from .weights import DIMENSIONS, diff as weights_diff, normalize

#: 相关性阈值 → 结论措辞
VERDICT_BANDS: tuple[tuple[float, str], ...] = (
    (0.98, "两次排序几乎完全一致"),
    (0.90, "两次排序高度一致，仅在个别商品上有出入"),
    (0.75, "两次排序有可观察的差异"),
    (0.50, "两次排序差异明显"),
    (0.00, "两次排序基本是两套不同的结论"),
)


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #

def average_ranks(values: Sequence[float], descending: bool = False) -> list[float]:
    """排名（并列取平均名次，从 1 开始）。

    Args:
        descending: ``True`` 表示数值越大名次越靠前（用于分数）。
    """
    n = len(values)
    if n == 0:
        return []
    sign = -1.0 if descending else 1.0
    order = sorted(range(n), key=lambda i: sign * values[i])
    ranks = [0.0] * n
    index = 0
    while index < n:
        end = index
        while end + 1 < n and values[order[end + 1]] == values[order[index]]:
            end += 1
        shared = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[order[position]] = shared
        index = end + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 1.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    var_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if var_x == 0 or var_y == 0:
        # 完全没有方差：两者相同则视为一致
        return 1.0 if list(xs) == list(ys) else 0.0
    return cov / (var_x * var_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """秩相关系数（对并列值取平均名次后再求 Pearson）。"""
    if len(xs) != len(ys):
        raise ValueError("spearman 要求两个序列等长")
    if not xs:
        return 1.0
    return round(_pearson(average_ranks(xs), average_ranks(ys)), 4)


def top_overlap(ranks_a: Sequence[int], ranks_b: Sequence[int], n: int) -> float:
    """Top-N 重合比例（0-1）。"""
    if n <= 0:
        return 0.0
    top_a = {index for index, rank in enumerate(ranks_a) if rank <= n}
    top_b = {index for index, rank in enumerate(ranks_b) if rank <= n}
    if not top_a and not top_b:
        return 1.0
    return round(len(top_a & top_b) / max(len(top_a), len(top_b), 1), 4)


def verdict_for(rho: float) -> str:
    for threshold, text in VERDICT_BANDS:
        if rho >= threshold:
            return text
    return VERDICT_BANDS[-1][1]


# --------------------------------------------------------------------------- #
# 两次运行对比
# --------------------------------------------------------------------------- #

@dataclass
class Mover:
    """排名变动最大的单个商品。"""

    product_id: int
    title: str
    category: str
    total_a: float
    total_b: float
    rank_a: int
    rank_b: int

    @property
    def rank_delta(self) -> int:
        """正数表示在新方案下排名上升。"""
        return self.rank_a - self.rank_b

    @property
    def score_delta(self) -> float:
        return round(self.total_b - self.total_a, 2)

    @property
    def direction(self) -> str:
        if self.rank_delta > 0:
            return "上升"
        if self.rank_delta < 0:
            return "下降"
        return "持平"


@dataclass
class ComparisonResult:
    """两次运行（或两套权重）的对比结果。"""

    run_a: dict[str, Any]
    run_b: dict[str, Any]
    common: int
    only_a: int
    only_b: int
    spearman: float
    avg_abs_rank_delta: float
    max_rank_delta: int
    top_overlap: dict[int, float]
    movers: list[Mover]
    weight_diff: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return verdict_for(self.spearman)

    def summary(self) -> str:
        parts = [
            f"共同商品 {self.common} 个",
            f"Spearman ρ = {self.spearman:.4f}（{self.verdict}）",
            f"平均排名变动 {self.avg_abs_rank_delta:.1f} 位，最大 {self.max_rank_delta} 位",
        ]
        overlaps = "、".join(f"Top{n} {value:.0%}" for n, value in self.top_overlap.items())
        if overlaps:
            parts.append(overlaps)
        return "；".join(parts) + "。"


def _index_by_product(items: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(item["product_id"]): item for item in items}


def compare_runs(
    run_a: Mapping[str, Any],
    items_a: Sequence[Mapping[str, Any]],
    run_b: Mapping[str, Any],
    items_b: Sequence[Mapping[str, Any]],
    *,
    top_n: Sequence[int] = (5, 10, 20),
    movers: int = 10,
) -> ComparisonResult:
    """对比两次打分运行。

    只统计**两次都出现**的商品（交集），避免新增/下架商品污染相关性。
    """
    map_a, map_b = _index_by_product(items_a), _index_by_product(items_b)
    common = [pid for pid in map_a if pid in map_b]
    common.sort(key=lambda pid: map_a[pid]["rank_no"])

    notes: list[str] = []
    only_a, only_b = len(map_a) - len(common), len(map_b) - len(common)
    if only_a or only_b:
        notes.append(
            f"仅出现在 A 的 {only_a} 个、仅出现在 B 的 {only_b} 个商品已排除在相关性计算之外。"
        )

    if not common:
        return ComparisonResult(
            run_a=dict(run_a), run_b=dict(run_b), common=0, only_a=only_a, only_b=only_b,
            spearman=0.0, avg_abs_rank_delta=0.0, max_rank_delta=0,
            top_overlap={n: 0.0 for n in top_n}, movers=[], notes=notes,
        )

    totals_a = [float(map_a[pid]["total"]) for pid in common]
    totals_b = [float(map_b[pid]["total"]) for pid in common]
    ranks_a = [int(map_a[pid]["rank_no"]) for pid in common]
    ranks_b = [int(map_b[pid]["rank_no"]) for pid in common]

    deltas = [abs(a - b) for a, b in zip(ranks_a, ranks_b)]
    overlaps = {n: top_overlap(ranks_a, ranks_b, n) for n in top_n}

    ranked_movers: list[Mover] = []
    for position, pid in enumerate(common):
        item_a, item_b = map_a[pid], map_b[pid]
        ranked_movers.append(Mover(
            product_id=pid,
            title=str(item_b.get("title") or item_a.get("title") or ""),
            category=str(item_b.get("category") or item_a.get("category") or ""),
            total_a=round(totals_a[position], 2),
            total_b=round(totals_b[position], 2),
            rank_a=ranks_a[position],
            rank_b=ranks_b[position],
        ))
    ranked_movers.sort(key=lambda mover: (-abs(mover.rank_delta), -abs(mover.score_delta)))

    weights_a = normalize(run_a.get("weights") or {})
    weights_b = normalize(run_b.get("weights") or {})
    if weights_a != weights_b:
        notes.append("两次运行使用了不同的权重方案。")

    return ComparisonResult(
        run_a=dict(run_a), run_b=dict(run_b), common=len(common),
        only_a=only_a, only_b=only_b,
        spearman=spearman(totals_a, totals_b),
        avg_abs_rank_delta=round(sum(deltas) / len(deltas), 2),
        max_rank_delta=max(deltas),
        top_overlap=overlaps,
        movers=ranked_movers[:movers],
        weight_diff=weights_diff(weights_a, weights_b),
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# 维度敏感度
# --------------------------------------------------------------------------- #

@dataclass
class DimensionImpact:
    """某维度对当前排序的影响力。"""

    dimension: str
    label: str
    spearman: float
    max_rank_delta: int
    avg_abs_rank_delta: float
    top_overlap: float
    weight: float

    @property
    def influence(self) -> float:
        """影响力 0-1：1 - ρ。ρ 越高说明去掉它几乎不影响排序。"""
        return round(1.0 - self.spearman, 4)

    @property
    def verdict(self) -> str:
        if self.spearman >= 0.99:
            return "几乎不影响排序"
        if self.spearman >= 0.95:
            return "影响很小"
        if self.spearman >= 0.85:
            return "有中等影响"
        return "对排序起决定作用"


def weight_sensitivity(
    items: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    *,
    top_n: int = 10,
) -> list[DimensionImpact]:
    """逐维度归零后重算排名，衡量该维度对当前排序的影响力。

    需要每个条目带有 ``dimensions`` 字段（快照里已经存了）。
    返回按影响力倒序排列的结果。
    """
    dimensions = [dict(item.get("dimensions") or {}) for item in items]
    if not dimensions:
        return []

    normalized = normalize(weights)
    baseline = [weighted_total(dims, normalized) for dims in dimensions]
    base_ranks = average_ranks(baseline, descending=True)

    impacts: list[DimensionImpact] = []
    for name in DIMENSIONS:
        if normalized.get(name, 0.0) <= 0:
            continue
        reduced = dict(normalized)
        reduced[name] = 0.0
        if sum(reduced.values()) <= 0:
            continue
        scores = [weighted_total(dims, reduced) for dims in dimensions]
        ranks = average_ranks(scores, descending=True)
        deltas = [abs(a - b) for a, b in zip(base_ranks, ranks)]
        impacts.append(DimensionImpact(
            dimension=name,
            label=DIMENSION_LABELS.get(name, name),
            spearman=spearman(baseline, scores),
            max_rank_delta=int(max(deltas)) if deltas else 0,
            avg_abs_rank_delta=round(sum(deltas) / len(deltas), 2) if deltas else 0.0,
            top_overlap=top_overlap(base_ranks, ranks, top_n),
            weight=round(normalized.get(name, 0.0), 4),
        ))

    impacts.sort(key=lambda impact: impact.influence, reverse=True)
    return impacts


def top_contributors(item: Mapping[str, Any], weights: Mapping[str, float],
                     limit: int = 3) -> list[dict[str, Any]]:
    """某个商品的总分主要来自哪些维度（按加权贡献倒序）。"""
    dims = dict(item.get("dimensions") or {})
    normalized = normalize(weights)
    total_weight = sum(normalized.get(name, 0.0) for name in dims) or 1.0
    rows = []
    for name, value in dims.items():
        share = normalized.get(name, 0.0) / total_weight
        rows.append({
            "dimension": name,
            "label": DIMENSION_LABELS.get(name, name),
            "score": round(float(value), 1),
            "weight": round(normalized.get(name, 0.0), 4),
            "contribution": round(float(value) * share, 2),
        })
    rows.sort(key=lambda row: row["contribution"], reverse=True)
    return rows[:limit]
