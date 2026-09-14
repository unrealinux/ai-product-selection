"""打分引擎单元测试。

    pytest -q
"""

from __future__ import annotations

import pytest

from app.models import ProductIn
from app.scoring import (
    build_advice,
    clamp,
    grade_of,
    linear_map,
    profit_margin,
    rule_score,
    score_dimensions,
    weighted_total,
)


def make_product(**overrides) -> ProductIn:
    base = dict(
        title="测试商品",
        category="测试",
        price=100.0,
        cost=30.0,
        heat=60.0,
        competition=40.0,
        weight_kg=0.5,
        repurchase=50.0,
        compliance_risk=10.0,
        virality=50.0,
    )
    base.update(overrides)
    return ProductIn(**base)


@pytest.mark.parametrize(
    "value,expected",
    [(-10, 0.0), (0, 0.0), (50, 50.0), (100, 100.0), (150, 100.0)],
)
def test_clamp(value, expected):
    assert clamp(value) == expected


def test_profit_margin():
    assert profit_margin(100, 30) == pytest.approx(0.7)
    assert profit_margin(100, 100) == 0.0
    assert profit_margin(0, 10) == 0.0


def test_linear_map_normal_and_reverse():
    assert linear_map(0.5, 0.0, 1.0) == pytest.approx(50.0)
    # 反向映射：重量越大分越低
    assert linear_map(0.1, 1.0, 5.0, 100.0, 0.0) == pytest.approx(100.0)
    assert linear_map(5.0, 1.0, 5.0, 100.0, 0.0) == pytest.approx(0.0)
    assert linear_map(9.0, 1.0, 5.0, 100.0, 0.0) == pytest.approx(0.0)


def test_dimensions_are_inverse_for_competition_and_risk():
    product = make_product(competition=80, compliance_risk=60)
    dims = score_dimensions(product)
    assert dims["competition"] == pytest.approx(20.0)
    assert dims["compliance"] == pytest.approx(40.0)
    assert all(0 <= value <= 100 for value in dims.values())


def test_weighted_total_with_uniform_dimensions():
    dims = {key: 80.0 for key in
            ["demand", "competition", "margin", "shipping", "repurchase", "compliance", "virality"]}
    assert weighted_total(dims) == pytest.approx(80.0, abs=0.01)


def test_weighted_total_normalizes_weights():
    assert weighted_total({"a": 100.0, "b": 0.0}, {"a": 3.0, "b": 1.0}) == pytest.approx(75.0)


@pytest.mark.parametrize(
    "total,expected",
    [(95, "S"), (85, "S"), (80, "A"), (70, "B"), (60, "C"), (40, "D")],
)
def test_grade_boundaries(total, expected):
    assert grade_of(total) == expected


def test_high_quality_product_scores_higher():
    good = make_product(price=199, cost=50, heat=90, competition=20,
                        weight_kg=0.3, repurchase=80, compliance_risk=5, virality=85)
    bad = make_product(price=199, cost=190, heat=20, competition=90,
                       weight_kg=4.5, repurchase=10, compliance_risk=80, virality=10)
    good_result, bad_result = rule_score(good), rule_score(bad)
    assert good_result.total > bad_result.total + 40
    assert good_result.grade in {"S", "A"}
    assert bad_result.grade == "D"


def test_advice_mentions_key_risks():
    product = make_product(price=100, cost=95, competition=85, weight_kg=4.0,
                           compliance_risk=70, heat=30)
    advice = build_advice(product, score_dimensions(product), 30.0)
    assert "毛利率" in advice
    assert "竞争度" in advice
    assert "合规" in advice
    assert advice.startswith("【")
