"""效果回测：把「打分排序」和「真实经营结果」接起来。

为什么需要它
------------
``compare.py`` 只能回答「换一套权重，排序变不变」，回答不了**「哪套权重更赚钱」**。
两套权重可能给出完全不同的排序，但谁对谁错，只有真实结果能判定。

本模块用**相关系数 + Top vs 其余的分组对比**做这件事：

* :func:`aggregate_outcomes` —— 把某个商品的多段经营数据按原始量求和，再推导比率
* :func:`backtest` —— 把一次打分快照的分数/排名与结果指标求 Spearman ρ，
  并比较「Top 组」与「其余」的指标差距，得到可读结论
* :func:`compare_backtests` —— 两套权重谁的分数更能预测结果

.. warning::
   这是**相关性**，不是因果。销量高的商品可能只是因为本身需求大，而不是因为
   打分高。要接近因果，需要把商品随机分组做 A/B。本模块的结论只能用于
   **筛掉明显帮倒忙的权重**，不能当成「提高了 x% 利润」的证据。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

from .compare import spearman
from .tabular import parse_number

#: 从各种写法里提取年月日（2024-03-01 / 2024/3/1 / 2024.3.1 / 2024年3月1日）
_DAY_RE = re.compile(r"(\d{4})\D{1,2}(\d{1,2})\D{1,2}(\d{1,2})")


def parse_day(value: Any) -> str | None:
    """把日期统一成 ``YYYY-MM-DD``；无法解析时返回 ``None``。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    match = _DAY_RE.search(str(value))
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None

# --------------------------------------------------------------------------- #
# 指标定义
# --------------------------------------------------------------------------- #

#: 每个指标：中文名、是否越大越好、格式化方式。``higher_better=False`` 表示越小越好
#: （如退货率），backtest 的结论方向会据此翻转。
METRICS: dict[str, dict[str, Any]] = {
    "gross_profit": {"label": "毛利额", "higher_better": True, "unit": "元", "digits": 1},
    "revenue": {"label": "成交金额", "higher_better": True, "unit": "元", "digits": 1},
    "orders": {"label": "订单数", "higher_better": True, "unit": "", "digits": 1},
    "units": {"label": "销量", "higher_better": True, "unit": "件", "digits": 1},
    "gross_margin": {"label": "实际毛利率", "higher_better": True, "unit": "", "digits": 4},
    "roi": {"label": "投产比", "higher_better": True, "unit": "", "digits": 2},
    "cvr": {"label": "转化率", "higher_better": True, "unit": "", "digits": 4},
    "ctr": {"label": "点击率", "higher_better": True, "unit": "", "digits": 4},
    "return_rate": {"label": "退货率", "higher_better": False, "unit": "", "digits": 4},
}

DEFAULT_METRIC = "gross_profit"

#: 少于这个样本量时结论不可靠，只做提示
MIN_SAMPLE = 5

#: 结果表里的原始计数字段（求和用）
COUNT_FIELDS: tuple[str, ...] = (
    "impressions", "clicks", "orders", "units", "returns",
)
#: 金额字段（求和用）
MONEY_FIELDS: tuple[str, ...] = ("revenue", "cogs", "ad_spend")


def metric_label(name: str) -> str:
    return METRICS.get(name, {}).get("label", name)


def metric_higher_better(name: str) -> bool:
    return bool(METRICS.get(name, {}).get("higher_better", True))


def format_metric(name: str, value: float | None) -> str:
    """按指标量级格式化，用于命令行与界面展示。"""
    if value is None:
        return "—"
    spec = METRICS.get(name, {})
    digits = spec.get("digits", 2)
    text = f"{value:.{digits}f}" if digits else f"{value:.0f}"
    if spec.get("unit") == "元":
        return f"{text} 元"
    if digits >= 4:
        return f"{value:.1%}"  # 比率类：转换成百分比更好读
    return text


# --------------------------------------------------------------------------- #
# 单商品的经营结果
# --------------------------------------------------------------------------- #

@dataclass
class OutcomeMetrics:
    """把若干段经营数据合并后的原始量，以及由它们推导出的比率。"""

    impressions: float = 0.0
    clicks: float = 0.0
    orders: float = 0.0
    units: float = 0.0
    returns: float = 0.0
    revenue: float = 0.0
    cogs: float = 0.0
    ad_spend: float = 0.0

    @property
    def ctr(self) -> float:
        return round(self.clicks / self.impressions, 6) if self.impressions > 0 else 0.0

    @property
    def cvr(self) -> float:
        return round(self.orders / self.clicks, 6) if self.clicks > 0 else 0.0

    @property
    def return_rate(self) -> float:
        return round(self.returns / self.orders, 6) if self.orders > 0 else 0.0

    @property
    def gross_profit(self) -> float:
        """毛利额 = 成交金额 − 采购成本 − 推广花费。"""
        return round(self.revenue - self.cogs - self.ad_spend, 4)

    @property
    def gross_margin(self) -> float:
        return round(self.gross_profit / self.revenue, 6) if self.revenue > 0 else 0.0

    @property
    def roi(self) -> float:
        """投产比。未投广告时为 0 —— 注意这不是「不赚钱」，而是无法计算。"""
        return round(self.gross_profit / self.ad_spend, 6) if self.ad_spend > 0 else 0.0

    def value(self, metric: str) -> float:
        if metric not in METRICS:
            raise KeyError(f"未知指标 {metric!r}，可选：{', '.join(METRICS)}")
        if metric in COUNT_FIELDS or metric in MONEY_FIELDS:
            return float(getattr(self, metric))
        return float(getattr(self, metric))

    def as_dict(self) -> dict[str, float]:
        return {
            "impressions": self.impressions,
            "clicks": self.clicks,
            "orders": self.orders,
            "units": self.units,
            "returns": self.returns,
            "revenue": self.revenue,
            "cogs": self.cogs,
            "ad_spend": self.ad_spend,
            "ctr": self.ctr,
            "cvr": self.cvr,
            "return_rate": self.return_rate,
            "gross_profit": self.gross_profit,
            "gross_margin": self.gross_margin,
            "roi": self.roi,
        }


def _number(row: Mapping[str, Any], key: str) -> float:
    """容忍 None / 空串 / 千分位 / 货币符号 —— 报表里这几种写法都很常见。"""
    raw = row.get(key, 0)
    if raw is None or raw == "":
        return 0.0
    value = parse_number(raw)
    return float(value) if value is not None else 0.0


def aggregate_outcomes(rows: Iterable[Mapping[str, Any]]) -> OutcomeMetrics:
    """把同一商品的多段结果按原始量求和，再推导比率。

    关键点：**先加原始量再算比率**。如果先算每段的转化率再平均，
    会把「曝光 10 次」和「曝光 10 万次」的时段等权，结果是错的。
    """
    metrics = OutcomeMetrics()
    for row in rows:
        for key in (*COUNT_FIELDS, *MONEY_FIELDS):
            setattr(metrics, key, getattr(metrics, key) + _number(row, key))
    return metrics


def group_by_product(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            product_id = int(row["product_id"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(product_id, []).append(row)
    return grouped


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #

@dataclass
class BacktestRow:
    """回测里的一行：一次打分 + 该商品的真实结果。"""

    product_id: int
    title: str
    score: float
    rank: int
    metrics: OutcomeMetrics

    def value(self, metric: str) -> float:
        return self.metrics.value(metric)


@dataclass
class BacktestResult:
    """一次快照的回测结果。"""

    run_id: int | None
    run_label: str
    metric: str
    rows: list[BacktestRow] = field(default_factory=list)
    top_ratio: float = 0.3
    rho: float | None = None
    top_n: int = 0
    top_avg: float | None = None
    rest_avg: float | None = None
    lift: float | None = None
    #: 因结果窗口全部早于快照而被排除的商品数（``after_run_only`` 时才发生）
    excluded_before_run: int = 0
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def sample_size(self) -> int:
        return len(self.rows)

    @property
    def signed_rho(self) -> float | None:
        """按指标方向校正过的相关性：越大越好，>0 表示打分方向正确。"""
        if self.rho is None:
            return None
        return self.rho if metric_higher_better(self.metric) else -self.rho

    def summary(self) -> str:
        if not self.rows:
            return "没有任何商品同时具备打分与结果数据，无法回测。"
        parts = [
            f"快照「{self.run_label}」× 指标「{metric_label(self.metric)}」："
            f"可比商品 {self.sample_size} 个"
        ]
        if self.rho is not None:
            parts.append(f"Spearman ρ = {self.rho:.4f}")
        if self.top_avg is not None and self.rest_avg is not None:
            parts.append(
                f"Top{self.top_n} 均值 {format_metric(self.metric, self.top_avg)}"
                f" vs 其余 {format_metric(self.metric, self.rest_avg)}"
            )
        if self.lift is not None:
            parts.append(f"倍差 {self.lift:.2f}×")
        parts.append(self.verdict)
        return "；".join(parts) + "。"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_label": self.run_label,
            "metric": self.metric,
            "metric_label": metric_label(self.metric),
            "sample_size": self.sample_size,
            "top_ratio": self.top_ratio,
            "excluded_before_run": self.excluded_before_run,
            "top_n": self.top_n,
            "rho": self.rho,
            "signed_rho": self.signed_rho,
            "top_avg": self.top_avg,
            "rest_avg": self.rest_avg,
            "lift": self.lift,
            "verdict": self.verdict,
            "summary": self.summary(),
            "notes": list(self.notes),
            "rows": [
                {
                    "product_id": row.product_id,
                    "title": row.title,
                    "score": row.score,
                    "rank": row.rank,
                    "value": row.value(self.metric),
                    "metrics": row.metrics.as_dict(),
                }
                for row in self.rows
            ],
        }


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _verdict_for(result: BacktestResult) -> str:
    if result.sample_size < MIN_SAMPLE:
        return "样本不足（少于 5 个可比商品），结论仅供参考"
    signed = result.signed_rho
    if signed is None:
        return "结果数据无法计算相关性"
    if signed <= -0.30:
        return f"打分与结果反向（ρ={signed:.2f}）—— 当前权重可能在帮倒忙"
    if signed >= 0.50:
        if result.lift is not None and result.lift >= 1.2:
            return f"打分与结果同向且 Top 组明显更好（ρ={signed:.2f}）—— 权重可用"
        return f"打分与结果同向（ρ={signed:.2f}）—— 权重方向正确但区分度有限"
    return f"打分与结果关联很弱（ρ={signed:.2f}）—— 权重区分度不足或样本太杂"


def _window_start_day(row: Mapping[str, Any]) -> Optional[str]:
    """结果的窗口开始日（统一成 YYYY-MM-DD）；解析不出返回 None。"""
    return parse_day(row.get("window_start"))


def backtest(
    run: Mapping[str, Any],
    items: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
    metric: str = DEFAULT_METRIC,
    top_ratio: float = 0.3,
    *,
    after_run_only: bool = False,
) -> BacktestResult:
    """把一次打分快照与经营结果对照。

    Args:
        run: 快照记录（至少含 ``id`` / ``label``；有 ``created_at`` 时做时间校验）。
        items: 快照条目（含 ``product_id`` / ``total`` / ``rank_no`` / ``title``）。
        outcomes: 结果记录，每条含 ``product_id`` 与原始量字段。
        metric: :data:`METRICS` 中的指标名。
        top_ratio: Top 组占可比样本的比例（0-1）。
        after_run_only: 只采用**开始于快照创建之后**的结果窗口。
            默认为 False，但只要有窗口早于快照就会提示 —— 打分发生在结果之后
            就不构成预测，相关性没有意义。

    Returns:
        :class:`BacktestResult`；没有共同商品时 ``rows`` 为空，不抛异常。
    """
    if metric not in METRICS:
        raise KeyError(f"未知指标 {metric!r}，可选：{', '.join(METRICS)}")
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio 必须在 (0, 1] 区间")

    grouped = group_by_product(outcomes)
    run_day = parse_day(run.get("created_at")) if run.get("created_at") else None

    # 一个商品可能出现在多行 item 里，取第一条
    item_by_product: dict[int, Mapping[str, Any]] = {}
    for item in items:
        try:
            product_id = int(item["product_id"])
        except (KeyError, TypeError, ValueError):
            continue
        item_by_product.setdefault(product_id, item)

    rows: list[BacktestRow] = []
    predating_products = 0
    dropped_before_run: set[int] = set()
    for product_id, item in item_by_product.items():
        raw_rows = grouped.get(product_id)
        if not raw_rows:
            continue  # 没有结果数据的商品不参与回测，而不是拿 0 顶替

        if run_day:
            early = [row for row in raw_rows if (_window_start_day(row) or run_day) < run_day]
            if early:
                predating_products += 1
                if after_run_only:
                    raw_rows = [row for row in raw_rows if row not in early]
                    if not raw_rows:
                        dropped_before_run.add(product_id)
                        continue

        rows.append(BacktestRow(
            product_id=product_id,
            title=str(item.get("title") or ""),
            score=float(item.get("total") or 0.0),
            rank=int(item.get("rank_no") or len(rows) + 1),
            metrics=aggregate_outcomes(raw_rows),
        ))

    result = BacktestResult(
        run_id=run.get("id"),
        run_label=str(run.get("label") or ""),
        metric=metric,
        rows=rows,
        top_ratio=top_ratio,
        excluded_before_run=len(dropped_before_run),
    )
    if not rows:
        result.verdict = "没有可比数据"
        result.notes.append(
            "需要先为这些商品录入经营结果（scripts/record_outcome.py 或 POST /outcomes）。"
        )
        return result

    if run_day and predating_products:
        if after_run_only:
            result.notes.append(
                f"已只采用快照创建（{run_day}）之后的结果窗口，"
                f"排除 {result.excluded_before_run} 个商品。"
            )
        else:
            result.notes.append(
                f"{predating_products} 个商品的结果窗口早于快照创建时间（{run_day}）—— "
                "打分发生在结果之后就不构成预测。建议加 --after-run-only 只看快照之后的数据。"
            )

    values = [row.value(metric) for row in rows]
    scores = [row.score for row in rows]
    result.rho = spearman(scores, values)

    ordered = sorted(rows, key=lambda row: row.score, reverse=True)
    result.top_n = max(1, min(len(ordered), round(len(ordered) * top_ratio)))
    top = ordered[:result.top_n]
    rest = ordered[result.top_n:]
    result.top_avg = _mean([row.value(metric) for row in top])
    result.rest_avg = _mean([row.value(metric) for row in rest]) if rest else None

    if result.top_avg is not None and result.rest_avg:
        result.lift = round(result.top_avg / result.rest_avg, 4)
    elif result.top_avg is not None and result.rest_avg == 0:
        result.lift = None
        result.notes.append("其余组均值为 0，倍差无法计算。")

    if len(set(values)) == 1:
        result.notes.append(
            f"所有商品的「{metric_label(metric)}」完全相同，该指标没有区分度，"
            "相关性无意义。"
        )
    if result.sample_size < MIN_SAMPLE:
        result.notes.append(f"只有 {result.sample_size} 个可比商品，样本偏少。")
    result.notes.append("这是相关性而非因果：结果好坏可能来自商品本身，而不是打分。")

    result.verdict = _verdict_for(result)
    return result


# --------------------------------------------------------------------------- #
# 两套权重的回测对比
# --------------------------------------------------------------------------- #

@dataclass
class BacktestComparison:
    """同一指标下，两次快照谁更能预测结果。"""

    metric: str
    left: BacktestResult
    right: BacktestResult
    verdict: str = ""

    @property
    def better(self) -> str:
        """返回 ``"left"`` / ``"right"`` / ``"tie"`` / ``"none"``。"""
        left, right = self.left.signed_rho, self.right.signed_rho
        if left is None or right is None:
            return "none"
        if math.isclose(left, right, abs_tol=0.05):
            return "tie"
        return "left" if left > right else "right"

    def summary(self) -> str:
        if self.better == "none":
            return "两次快照中至少有一次没有可比数据，无法对比。"
        if self.better == "tie":
            return (
                f"两次快照对「{metric_label(self.metric)}」的预测力接近"
                f"（ρ {self.left.signed_rho:.3f} vs {self.right.signed_rho:.3f}），"
                "换权重没有明显收益。"
            )
        winner = self.left if self.better == "left" else self.right
        loser = self.right if self.better == "left" else self.left
        return (
            f"「{winner.run_label}」更能预测「{metric_label(self.metric)}」："
            f"ρ {winner.signed_rho:.3f} vs {loser.signed_rho:.3f}。"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "metric_label": metric_label(self.metric),
            "better": self.better,
            "verdict": self.verdict or self.summary(),
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
        }


def compare_backtests(left: BacktestResult, right: BacktestResult) -> BacktestComparison:
    """对比两次快照在同一指标上的预测力（按方向校正后的 ρ）。"""
    if left.metric != right.metric:
        raise ValueError("两次回测必须使用同一指标才能对比")
    comparison = BacktestComparison(metric=left.metric, left=left, right=right)
    comparison.verdict = comparison.summary()
    return comparison
