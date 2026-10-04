"""效果回测纯逻辑测试。

盯住三件容易算错、而且算错了不会报错的事：

1. **先加原始量再算比率** —— 若先算每段转化率再平均，会把小样本时段等权，结论失真
2. **方向** —— 退货率越小越好，直接拿 ρ 判好坏会得出相反结论
3. **样本与区分度** —— 指标全相同、样本过少时，ρ 没有意义，必须显式提示而不是给个数字
"""

from __future__ import annotations

import pytest

from app.outcomes import (
    METRICS,
    MIN_SAMPLE,
    BacktestResult,
    BacktestRow,
    OutcomeMetrics,
    aggregate_outcomes,
    backtest,
    compare_backtests,
    format_metric,
    group_by_product,
    metric_higher_better,
    metric_label,
)


# --------------------------------------------------------------------------- #
# 造数据
# --------------------------------------------------------------------------- #

def outcome(product_id: int, **kwargs) -> dict:
    row = {
        "product_id": product_id,
        "impressions": 0.0, "clicks": 0.0, "orders": 0.0, "units": 0.0,
        "returns": 0.0, "revenue": 0.0, "cogs": 0.0, "ad_spend": 0.0,
    }
    row.update(kwargs)
    return row


def run_stub(run_id: int = 1, label: str = "基线") -> dict:
    return {"id": run_id, "label": label}


def items_for(scores: list[float], titles: list[str] | None = None) -> list[dict]:
    titles = titles or [f"P{index}" for index in range(1, len(scores) + 1)]
    return [
        {"product_id": index + 1, "total": score, "rank_no": index + 1, "title": title}
        for index, (score, title) in enumerate(zip(scores, titles))
    ]


# --------------------------------------------------------------------------- #
# 聚合与推导
# --------------------------------------------------------------------------- #

def test_aggregate_sums_raw_values_before_deriving_ratios():
    """两段数据的点击/订单都不同：必须先加总再算转化率。"""
    rows = [
        outcome(1, impressions=100, clicks=10, orders=1, revenue=100.0, cogs=40.0),
        outcome(1, impressions=900, clicks=90, orders=18, revenue=900.0, cogs=360.0),
    ]

    metrics = aggregate_outcomes(rows)

    assert metrics.impressions == 1000
    assert metrics.clicks == 100
    assert metrics.orders == 19
    # 正确：19/100 = 0.19；若先算每段比率再平均会得到 (0.1+0.2)/2 = 0.15
    assert metrics.cvr == pytest.approx(0.19)
    assert metrics.ctr == pytest.approx(0.1)
    assert metrics.gross_profit == pytest.approx(600.0)
    assert metrics.gross_margin == pytest.approx(0.6)


def test_rates_are_zero_when_denominator_missing():
    metrics = OutcomeMetrics()
    assert metrics.ctr == 0.0
    assert metrics.cvr == 0.0
    assert metrics.return_rate == 0.0
    assert metrics.gross_margin == 0.0
    assert metrics.roi == 0.0  # 没投广告时无法计算，不是「不赚钱」


def test_aggregate_tolerates_strings_and_blanks():
    rows = [
        {"product_id": 1, "impressions": "1,000", "clicks": None, "revenue": "",
         "orders": "12"},
        {"product_id": 1, "impressions": "500", "clicks": "—", "revenue": "3,000.5"},
    ]

    metrics = aggregate_outcomes(rows)

    assert metrics.impressions == 1500.0
    assert metrics.clicks == 0.0
    assert metrics.revenue == pytest.approx(3000.5)
    assert metrics.orders == 12.0


def test_roi_uses_ad_spend():
    metrics = OutcomeMetrics(revenue=1000.0, cogs=400.0, ad_spend=200.0)
    assert metrics.gross_profit == pytest.approx(400.0)
    assert metrics.roi == pytest.approx(2.0)


def test_value_rejects_unknown_metric():
    with pytest.raises(KeyError):
        OutcomeMetrics().value("不存在的指标")


def test_group_by_product_skips_invalid_rows():
    grouped = group_by_product([
        outcome(1), outcome(2), {"product_id": "x"}, {"no_id": 1},
    ])
    assert set(grouped) == {1, 2}


# --------------------------------------------------------------------------- #
# 相关性方向
# --------------------------------------------------------------------------- #

def test_backtest_detects_perfect_positive_correlation():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=1000 - index * 100) for index in range(1, 7)]

    result = backtest(run_stub(), items, outcomes, metric="gross_profit", top_ratio=0.3)

    assert result.sample_size == 6
    assert result.rho == pytest.approx(1.0)
    assert result.signed_rho == pytest.approx(1.0)
    assert result.top_n == 2
    assert result.top_avg == pytest.approx(850.0)
    assert result.rest_avg == pytest.approx(550.0)
    assert result.lift == pytest.approx(1.5455, abs=1e-3)
    assert "权重可用" in result.verdict


def test_backtest_flags_inverse_correlation():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=index * 100) for index in range(1, 7)]

    result = backtest(run_stub(), items, outcomes, metric="gross_profit")

    assert result.rho == pytest.approx(-1.0)
    assert result.signed_rho == pytest.approx(-1.0)
    assert "帮倒忙" in result.verdict


def test_backtest_flips_direction_for_lower_is_better_metric():
    """退货率越低越好：高分商品退货率低时，结论应是正面的。"""
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, orders=10, returns=index) for index in range(1, 7)]

    result = backtest(run_stub(), items, outcomes, metric="return_rate")

    assert result.rho == pytest.approx(-1.0)
    assert result.signed_rho == pytest.approx(1.0)  # 方向校正后为正
    assert "帮倒忙" not in result.verdict


def test_backtest_warns_when_metric_has_no_variance():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=500.0) for index in range(1, 7)]

    result = backtest(run_stub(), items, outcomes, metric="gross_profit")

    assert any("没有区分度" in note for note in result.notes)


def test_backtest_warns_on_small_sample():
    items = items_for([90, 80, 70])
    outcomes = [outcome(index, revenue=1000 - index * 100) for index in range(1, 4)]

    result = backtest(run_stub(), items, outcomes)

    assert result.sample_size < MIN_SAMPLE
    assert any("样本偏少" in note for note in result.notes)
    assert "样本不足" in result.verdict


def test_backtest_always_notes_correlation_is_not_causation():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=1000 - index * 100) for index in range(1, 7)]
    result = backtest(run_stub(), items, outcomes)
    assert any("而非因果" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# 边界
# --------------------------------------------------------------------------- #

def test_backtest_without_overlapping_products_returns_empty():
    items = items_for([90, 80])
    result = backtest(run_stub(), items, [], metric="gross_profit")

    assert result.rows == []
    assert result.verdict == "没有可比数据"
    assert result.notes
    assert "无法回测" in result.summary()


def test_backtest_skips_items_without_outcomes():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=500.0) for index in (1, 2, 3)]  # 只有前 3 个有结果

    result = backtest(run_stub(), items, outcomes)

    assert {row.product_id for row in result.rows} == {1, 2, 3}


def test_backtest_top_ratio_of_one_leaves_no_rest_group():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=1000 - index * 100) for index in range(1, 7)]

    result = backtest(run_stub(), items, outcomes, top_ratio=1.0)

    assert result.top_n == 6
    assert result.rest_avg is None
    assert result.lift is None


def test_backtest_rejects_bad_arguments():
    items = items_for([90, 80])
    with pytest.raises(KeyError):
        backtest(run_stub(), items, [], metric="不存在")
    with pytest.raises(ValueError):
        backtest(run_stub(), items, [], top_ratio=0)
    with pytest.raises(ValueError):
        backtest(run_stub(), items, [], top_ratio=1.5)


def test_backtest_result_serialises_to_dict():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=1000 - index * 100) for index in range(1, 7)]
    payload = backtest(run_stub(7, "基线"), items, outcomes).as_dict()

    assert payload["run_id"] == 7
    assert payload["metric_label"] == "毛利额"
    assert payload["sample_size"] == 6
    assert len(payload["rows"]) == 6
    assert set(payload["rows"][0]["metrics"]) >= {"ctr", "cvr", "gross_profit", "roi"}
    assert payload["summary"]


# --------------------------------------------------------------------------- #
# 两套权重对比
# --------------------------------------------------------------------------- #

def fake_result(label: str, rho: float, metric: str = "gross_profit",
                with_rows: bool = True) -> BacktestResult:
    result = BacktestResult(run_id=1, run_label=label, metric=metric)
    # 没有可比商品时真实回测根本算不出 rho，这里必须一致
    result.rho = rho if with_rows else None
    if with_rows:
        result.rows = [BacktestRow(product_id=1, title="A", score=90.0, rank=1,
                                   metrics=OutcomeMetrics(revenue=100.0))]
    return result


def test_compare_picks_the_better_predictor():
    comparison = compare_backtests(
        fake_result("均衡", 0.82), fake_result("毛利优先", 0.31)
    )
    assert comparison.better == "left"
    assert "均衡" in comparison.summary()


def test_compare_reports_tie_when_close():
    comparison = compare_backtests(
        fake_result("A", 0.60), fake_result("B", 0.62)
    )
    assert comparison.better == "tie"
    assert "没有明显收益" in comparison.summary()


def test_compare_without_data_returns_none():
    comparison = compare_backtests(
        fake_result("A", 0.5, with_rows=False), fake_result("B", 0.7)
    )
    assert comparison.better == "none"
    assert "无法对比" in comparison.summary()


def test_compare_requires_same_metric():
    with pytest.raises(ValueError):
        compare_backtests(
            fake_result("A", 0.5, metric="orders"),
            fake_result("B", 0.5, metric="revenue"),
        )


# --------------------------------------------------------------------------- #
# 时间校验：打分必须发生在结果之前
# --------------------------------------------------------------------------- #

def run_dated(created_at: str) -> dict:
    return {"id": 1, "label": "基线", "created_at": created_at}


def test_backtest_warns_when_outcomes_predate_snapshot():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=1000 - index * 100, window_start="2024-01-01")
                for index in range(1, 7)]

    result = backtest(run_dated("2024-03-01"), items, outcomes)

    assert result.sample_size == 6  # 默认仍然纳入
    assert any("早于快照" in note for note in result.notes)


def test_after_run_only_excludes_predating_products():
    items = items_for([90, 80, 70, 60, 50, 40])
    outcomes = [outcome(index, revenue=1000 - index * 100, window_start="2024-01-01")
                for index in range(1, 7)]

    result = backtest(run_dated("2024-03-01"), items, outcomes, after_run_only=True)

    assert result.rows == []
    assert result.excluded_before_run == 6
    assert result.verdict == "没有可比数据"


def test_after_run_only_keeps_only_later_windows():
    items = items_for([90, 80])
    outcomes = [
        outcome(1, revenue=1000.0, window_start="2024-01-01"),  # 早于快照
        outcome(1, revenue=500.0, window_start="2024-04-01"),   # 快照之后
        outcome(2, revenue=300.0, window_start="2024-04-01"),
    ]

    result = backtest(run_dated("2024-03-01"), items, outcomes, after_run_only=True)

    assert result.sample_size == 2
    by_id = {row.product_id: row for row in result.rows}
    # 商品 1 只统计快照之后那一段，不能把 1000 也加进来
    assert by_id[1].metrics.revenue == pytest.approx(500.0)
    assert result.excluded_before_run == 0


def test_no_temporal_note_without_created_at():
    items = items_for([90, 80])
    outcomes = [outcome(index, revenue=100.0, window_start="2024-01-01")
                for index in (1, 2)]
    result = backtest(run_stub(), items, outcomes)
    assert not any("早于快照" in note for note in result.notes)


def test_unparseable_window_is_not_treated_as_predating():
    items = items_for([90, 80])
    outcomes = [outcome(index, revenue=100.0, window_start="很久以前")
                for index in (1, 2)]
    result = backtest(run_dated("2024-03-01"), items, outcomes, after_run_only=True)
    assert result.sample_size == 2  # 解析不出时间就不武断排除


def test_excluded_before_run_is_serialised():
    items = items_for([90, 80])
    outcomes = [outcome(index, revenue=100.0, window_start="2024-01-01")
                for index in (1, 2)]
    payload = backtest(run_dated("2024-03-01"), items, outcomes,
                       after_run_only=True).as_dict()
    assert "excluded_before_run" in payload


# --------------------------------------------------------------------------- #
# 展示辅助
# --------------------------------------------------------------------------- #

def test_format_metric_matches_units():
    assert format_metric("gross_profit", 1234.5) == "1234.5 元"
    assert format_metric("orders", 3.0) == "3.0"
    assert format_metric("cvr", 0.2) == "20.0%"
    assert format_metric("cvr", None) == "—"


def test_metric_metadata():
    assert metric_label("gross_profit") == "毛利额"
    assert metric_label("unknown") == "unknown"
    assert metric_higher_better("return_rate") is False
    assert metric_higher_better("orders") is True
    assert set(METRICS) >= {"gross_profit", "return_rate", "roi", "cvr"}
