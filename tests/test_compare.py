"""权重方案与 A/B 对比测试。

重点覆盖排名数学：并列名次处理、Spearman 的已知值、Top-N 重合度。
这些地方算错了不会报错，只会给出看起来合理的错误结论。
"""

from __future__ import annotations

import pytest

from app import db
from app.compare import (
    DimensionImpact,
    Mover,
    average_ranks,
    compare_runs,
    spearman,
    top_contributors,
    top_overlap,
    verdict_for,
    weight_sensitivity,
)
from app.config import DEFAULT_WEIGHTS
from app.models import ProductIn
from app.weights import (
    DIMENSIONS,
    PRESETS,
    describe,
    diff,
    merge,
    normalize,
    parse_weights,
    preset_weights,
    validate,
)

# --------------------------------------------------------------------------- #
# 排名数学
# --------------------------------------------------------------------------- #

def test_average_ranks_without_ties():
    assert average_ranks([10, 30, 20]) == [1.0, 3.0, 2.0]


def test_average_ranks_averages_ties():
    # 1 和 1 并列第 1、2 名 → 都记 1.5；3 是第 3 名
    assert average_ranks([1, 1, 3]) == [1.5, 1.5, 3.0]


def test_average_ranks_descending_puts_largest_first():
    assert average_ranks([10, 30, 20], descending=True) == [3.0, 1.0, 2.0]


def test_average_ranks_empty():
    assert average_ranks([]) == []


def test_spearman_identical_is_one():
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_spearman_reversed_is_minus_one():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_spearman_known_value_without_ties():
    """经典公式：Σd² = 2, n = 3 → ρ = 1 - 12/24 = 0.5"""
    assert spearman([1, 2, 3], [1, 3, 2]) == pytest.approx(0.5)


def test_spearman_known_value_with_ties():
    """并列值取平均名次后再算 Pearson → 0.8660"""
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660, abs=1e-4)


def test_spearman_requires_equal_length():
    with pytest.raises(ValueError, match="等长"):
        spearman([1, 2], [1, 2, 3])


def test_spearman_constant_series():
    # 全等且完全一致 → 视为一致
    assert spearman([5, 5, 5], [5, 5, 5]) == 1.0
    # 一个没有方差、另一个有 → 视为不一致
    assert spearman([5, 5, 5], [1, 2, 3]) == 0.0


def test_top_overlap():
    assert top_overlap([1, 2, 3, 4], [1, 2, 5, 6], 2) == 1.0
    assert top_overlap([1, 2, 3, 4], [3, 4, 1, 2], 2) == 0.0
    assert top_overlap([1, 2, 3, 4], [1, 2, 3, 4], 10) == 1.0


def test_verdict_bands():
    assert "几乎完全一致" in verdict_for(1.0)
    assert "高度一致" in verdict_for(0.92)
    assert "可观察" in verdict_for(0.80)
    assert "明显" in verdict_for(0.60)
    assert "两套不同" in verdict_for(0.10)


# --------------------------------------------------------------------------- #
# 权重方案
# --------------------------------------------------------------------------- #

def test_presets_are_normalised_and_cover_all_dimensions():
    for name, spec in PRESETS.items():
        weights = spec["weights"]
        assert set(weights) == set(DIMENSIONS), name
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9), name
        assert all(value >= 0 for value in weights.values()), name


def test_preset_weights_returns_copy():
    first = preset_weights("balanced")
    first["margin"] = 99
    assert preset_weights("balanced")["margin"] != 99


def test_preset_unknown_name():
    with pytest.raises(KeyError, match="未知预设"):
        preset_weights("nope")


def test_normalize_sums_to_one_and_fills_missing():
    result = normalize({"margin": 3, "demand": 1})
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["margin"] == pytest.approx(0.75)
    assert result["demand"] == pytest.approx(0.25)
    assert result["shipping"] == 0.0
    assert set(result) == set(DIMENSIONS)


def test_normalize_drops_unknown_and_clamps_negative():
    result = normalize({"margin": 1, "demand": -5, "不存在的维度": 10})
    assert result["margin"] == pytest.approx(1.0)
    assert result["demand"] == 0.0
    assert "不存在的维度" not in result


def test_normalize_all_zero_falls_back_to_defaults():
    assert normalize({name: 0 for name in DIMENSIONS}) == dict(DEFAULT_WEIGHTS)
    assert normalize({}) == dict(DEFAULT_WEIGHTS)


def test_validate_accepts_good_weights():
    assert validate(DEFAULT_WEIGHTS) == []
    assert validate({"margin": 1}) == []


def test_validate_reports_problems():
    assert any("未知维度" in p for p in validate({"bogus": 1}))
    assert any("不能为负数" in p for p in validate({"margin": -1}))
    assert any("不是数字" in p for p in validate({"margin": "abc"}))
    assert any("至少要有一个" in p for p in validate({"margin": 0, "demand": 0}))


def test_parse_weights_english_and_chinese():
    assert parse_weights("margin=0.4,demand=0.2") == {"margin": 0.4, "demand": 0.2}
    # 中文必须用完整维度名（避免「毛利」这种缩写歧义）
    assert parse_weights("毛利率=0.4,需求热度=0.2") == {"margin": 0.4, "demand": 0.2}
    with pytest.raises(ValueError, match="未知维度"):
        parse_weights("毛利=0.4")  # 缩写不认


def test_parse_weights_handles_full_width_comma_and_spaces():
    assert parse_weights(" margin = 0.5 ， demand = 0.5 ") == {"margin": 0.5, "demand": 0.5}


def test_parse_weights_errors():
    with pytest.raises(ValueError, match="缺少 '='"):
        parse_weights("margin")
    with pytest.raises(ValueError, match="未知维度"):
        parse_weights("bogus=1")
    with pytest.raises(ValueError, match="不是数字"):
        parse_weights("margin=abc")


def test_parse_weights_empty():
    assert parse_weights("") == {}


def test_merge_layers_and_normalise():
    base = {"margin": 0.5, "demand": 0.5}
    merged = merge(base, {"margin": 0.9})
    # merge 会归一化，所以 margin 变成 0.9/(0.9+0.5)
    assert merged["margin"] == pytest.approx(0.9 / 1.4)
    assert merged["margin"] > base["margin"]
    assert sum(merged.values()) == pytest.approx(1.0)
    # 未知维度被忽略
    assert merge(base, {"bogus": 1}) == normalize(base)


def test_diff_sorted_by_absolute_change():
    rows = diff(DEFAULT_WEIGHTS, preset_weights("margin_first"))
    assert rows[0]["dimension"] == "margin"
    deltas = [abs(row["delta"]) for row in rows]
    assert deltas == sorted(deltas, reverse=True)


def test_describe_shows_top_dimensions():
    text = describe(preset_weights("margin_first"))
    assert text.startswith("毛利率")
    assert text.count(">") == 2  # 默认取前三


# --------------------------------------------------------------------------- #
# 运行对比
# --------------------------------------------------------------------------- #

def make_run(run_id=1, label="run", weights=None, **overrides):
    base = {"id": run_id, "label": label, "weights": weights or DEFAULT_WEIGHTS,
            "product_count": 0, "avg_score": 0.0, "profile_id": None,
            "note": "", "created_at": None}
    base.update(overrides)
    return base


def make_items(totals, *, prefix="商品", dimensions=None, ranks=None):
    items = []
    for index, total in enumerate(totals):
        items.append({
            "product_id": index + 1,
            "title": f"{prefix}{index + 1}",
            "category": "测试",
            "source": "test",
            "total": total,
            "rank_no": ranks[index] if ranks else None,
            "dimensions": (dimensions or {}).get(index) or {
                "demand": 50.0, "competition": 50.0, "margin": 50.0,
                "shipping": 50.0, "repurchase": 50.0, "compliance": 50.0, "virality": 50.0,
            },
        })
    # 未显式给排名时按分数倒序生成
    if ranks is None:
        order = sorted(range(len(totals)), key=lambda i: -totals[i])
        for position, index in enumerate(order, start=1):
            items[index]["rank_no"] = position
    return items


def test_compare_identical_runs_has_no_movement():
    items = make_items([90, 80, 70, 60])
    result = compare_runs(make_run(1), items, make_run(2), items)

    assert result.common == 4
    assert result.spearman == 1.0
    assert result.avg_abs_rank_delta == 0.0
    assert result.max_rank_delta == 0
    assert all(mover.rank_delta == 0 for mover in result.movers)
    assert "几乎完全一致" in result.verdict


def test_compare_detects_rank_swaps():
    items_a = make_items([90, 80, 70, 60])
    # 交换前两名的分数
    items_b = make_items([80, 90, 70, 60])
    result = compare_runs(make_run(1), items_a, make_run(2), items_b, top_n=(2,))

    # 第 1、2 名互换 → 每边的位次变动都是 1
    assert result.max_rank_delta == 1
    assert result.avg_abs_rank_delta == pytest.approx(0.5)
    top = result.movers[0]
    assert top.rank_delta != 0
    assert result.top_overlap[2] == 1.0  # 前两名还是那两个商品，只是换了位次
    assert result.spearman < 1.0


def test_mover_direction_and_deltas():
    items_a = make_items([90, 80, 70])
    items_b = make_items([70, 90, 80])
    result = compare_runs(make_run(1), items_a, make_run(2), items_b)

    climber = next(m for m in result.movers if m.product_id == 2)
    assert climber.rank_a == 2 and climber.rank_b == 1
    assert climber.rank_delta == 1
    assert climber.direction == "上升"
    assert climber.score_delta == pytest.approx(90 - 80)

    faller = next(m for m in result.movers if m.product_id == 1)
    # A: [90,80,70] → p1 第 1；B: [70,90,80] → p1 跌到第 3
    assert faller.rank_a == 1 and faller.rank_b == 3
    assert faller.rank_delta == -2
    assert faller.direction == "下降"


def test_compare_excludes_products_missing_from_either_side():
    items_a = make_items([90, 80, 70])
    items_b = make_items([90, 80])  # 少一个商品
    result = compare_runs(make_run(1), items_a, make_run(2), items_b)

    assert result.common == 2
    assert result.only_a == 1
    assert result.only_b == 0
    assert any("仅出现在 A" in note for note in result.notes)


def test_compare_with_empty_intersection():
    items_a = [dict(item, product_id=100 + item["product_id"]) for item in make_items([90, 80])]
    items_b = make_items([90, 80])
    result = compare_runs(make_run(1), items_a, make_run(2), items_b)

    assert result.common == 0
    assert result.spearman == 0.0
    assert result.movers == []
    assert result.top_overlap[10] == 0.0


def test_compare_notes_weight_change():
    items = make_items([90, 80, 70])
    result = compare_runs(
        make_run(1, weights=DEFAULT_WEIGHTS), items,
        make_run(2, weights=preset_weights("margin_first")), items,
    )
    assert any("不同的权重方案" in note for note in result.notes)
    assert result.weight_diff[0]["dimension"] == "margin"


def test_compare_summary_is_readable():
    items = make_items([90, 80, 70, 60])
    summary = compare_runs(make_run(1), items, make_run(2), items).summary()
    assert "共同商品 4 个" in summary
    assert "Spearman" in summary
    assert "Top5" in summary


# --------------------------------------------------------------------------- #
# 维度敏感度
# --------------------------------------------------------------------------- #

def test_sensitivity_zeroing_a_dimension_with_no_variance_changes_nothing():
    """所有商品在某维度上得分相同 → 去掉它排序不变（ρ=1）。"""
    dims = [
        {"demand": 50.0, "competition": score, "margin": 50.0, "shipping": 50.0,
         "repurchase": 50.0, "compliance": 50.0, "virality": 50.0}
        for score in (90.0, 70.0, 60.0, 40.0)
    ]
    items = make_items([0, 0, 0, 0], dimensions={i: d for i, d in enumerate(dims)})
    impacts = {impact.dimension: impact for impact in weight_sensitivity(items, DEFAULT_WEIGHTS)}

    assert impacts["demand"].spearman == 1.0
    assert impacts["demand"].influence == 0.0
    assert impacts["demand"].verdict == "几乎不影响排序"
    # 唯一有区分度的维度，去掉后名次必然乱
    assert impacts["competition"].spearman < 1.0
    assert impacts["competition"].influence > impacts["demand"].influence


def test_sensitivity_sorted_by_influence_descending():
    dims = [
        {"demand": d, "competition": 50.0, "margin": c, "shipping": 50.0,
         "repurchase": 50.0, "compliance": 50.0, "virality": 50.0}
        for d, c in ((90.0, 10.0), (70.0, 80.0), (60.0, 30.0), (40.0, 95.0))
    ]
    items = make_items([0, 0, 0, 0], dimensions={i: d for i, d in enumerate(dims)})
    impacts = weight_sensitivity(items, DEFAULT_WEIGHTS)

    influences = [impact.influence for impact in impacts]
    assert influences == sorted(influences, reverse=True)
    assert impacts[0].weight > 0


def test_sensitivity_skips_zero_weight_dimensions():
    items = make_items([0, 0, 0], dimensions={i: {
        "demand": 50.0, "competition": 50.0, "margin": 50.0, "shipping": 50.0,
        "repurchase": 50.0, "compliance": 50.0, "virality": 50.0,
    } for i in range(3)})
    weights = dict(DEFAULT_WEIGHTS, virality=0.0)
    impacts = weight_sensitivity(items, weights)
    assert "virality" not in {impact.dimension for impact in impacts}


def test_sensitivity_empty_items():
    assert weight_sensitivity([], DEFAULT_WEIGHTS) == []


def test_sensitivity_reports_top_overlap_and_max_delta():
    dims = [
        {"demand": d, "competition": c, "margin": 50.0, "shipping": 50.0,
         "repurchase": 50.0, "compliance": 50.0, "virality": 50.0}
        for d, c in ((95.0, 10.0), (80.0, 95.0), (60.0, 20.0), (30.0, 90.0))
    ]
    items = make_items([0, 0, 0, 0], dimensions={i: d for i, d in enumerate(dims)})
    impacts = weight_sensitivity(items, DEFAULT_WEIGHTS, top_n=2)
    for impact in impacts:
        assert 0.0 <= impact.top_overlap <= 1.0
        assert impact.max_rank_delta >= 0
        assert isinstance(impact, DimensionImpact)


def test_top_contributors_ranks_by_weighted_contribution():
    item = {
        "dimensions": {"demand": 90.0, "competition": 20.0, "margin": 100.0,
                       "shipping": 10.0, "repurchase": 10.0, "compliance": 10.0,
                       "virality": 10.0},
    }
    rows = top_contributors(item, DEFAULT_WEIGHTS, limit=3)
    assert rows[0]["dimension"] == "margin"      # 100 分 × 24%
    assert rows[0]["label"] == "毛利率"
    assert len(rows) == 3
    contributions = [row["contribution"] for row in rows]
    assert contributions == sorted(contributions, reverse=True)


# --------------------------------------------------------------------------- #
# 落库：方案与快照
# --------------------------------------------------------------------------- #

def test_profile_round_trip(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)

    saved = db.upsert_profile("测试方案", {"margin": 0.7, "demand": 0.3}, "备注", path)
    assert saved["id"] > 0
    assert saved["weights"] == {"margin": 0.7, "demand": 0.3}

    # 同名更新而非新增
    again = db.upsert_profile("测试方案", {"margin": 0.1, "demand": 0.9}, "改过", path)
    assert again["id"] == saved["id"]
    assert again["weights"]["margin"] == 0.1
    assert len(db.list_profiles(path)) == 1

    assert db.get_profile("测试方案", path)["description"] == "改过"
    assert db.get_profile(saved["id"], path)["name"] == "测试方案"
    assert db.get_profile("不存在", path) is None

    assert db.delete_profile("测试方案", path) is True
    assert db.delete_profile("测试方案", path) is False


def test_run_snapshot_round_trip(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    product = db.upsert_product(
        ProductIn(title="快照商品", price=100, cost=30), path
    )

    items = [{
        "product_id": product.id, "title": product.title, "category": product.category,
        "source": product.source, "total": 77.5,
        "dimensions": {"margin": 80.0, "demand": 60.0},
    }]
    run = db.create_run("快照A", {"margin": 1.0}, items, note="备注", db_path=path)

    assert run["product_count"] == 1
    assert run["avg_score"] == pytest.approx(77.5)
    assert run["weights"] == {"margin": 1.0}

    stored = db.get_run_items(run["id"], path)
    assert len(stored) == 1
    assert stored[0]["rank_no"] == 1
    assert stored[0]["dimensions"]["margin"] == 80.0

    assert [r["id"] for r in db.list_runs(db_path=path)] == [run["id"]]
    assert db.get_run(run["id"], path)["label"] == "快照A"
    assert db.get_run(999, path) is None

    assert db.delete_run(run["id"], path) is True
    assert db.get_run_items(run["id"], path) == []


def test_run_ranking_is_by_total_descending(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)

    ids = []
    for name in ("甲", "乙", "丙"):
        ids.append(db.upsert_product(
            ProductIn(title=name, price=100, cost=50), path
        ).id)

    items = [
        {"product_id": ids[0], "title": "甲", "total": 50.0, "dimensions": {}},
        {"product_id": ids[1], "title": "乙", "total": 90.0, "dimensions": {}},
        {"product_id": ids[2], "title": "丙", "total": 70.0, "dimensions": {}},
    ]
    run = db.create_run("排序", {"margin": 1.0}, items, db_path=path)
    stored = db.get_run_items(run["id"], path)

    assert [row["rank_no"] for row in stored] == [1, 2, 3]
    assert [row["total"] for row in stored] == [90.0, 70.0, 50.0]
    assert stored[0]["title"] == "乙"
