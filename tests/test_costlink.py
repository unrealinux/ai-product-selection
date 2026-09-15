"""成本对齐测试。

盯住三类会静默出错的地方：

1. **相似度函数**：营销词造成假相似、规格差异被淹没、长度差异被忽略
2. **匹配策略**：不覆盖已有成本、低置信不默认应用、供货池为空时的行为
3. **负毛利诊断**：批发价高于零售价通常是匹配错了，不能当"亏本货"处理
"""

from __future__ import annotations

import pytest

from app import db
from app.costlink import (
    COST_SEP,
    DEFAULT_THRESHOLD,
    CostMatch,
    apply_matches,
    bigrams,
    cost_note,
    dice,
    jaccard,
    link_costs,
    margin_report,
    normalize_title,
    pick_supply,
    shared_specs,
    spec_tokens,
    title_similarity,
    with_cost,
)
from app.models import ProductIn

# --------------------------------------------------------------------------- #
# 造数据
# --------------------------------------------------------------------------- #

def make_product(title: str, *, price: float = 100.0, cost: float = 0.0,
                 source: str = "taobao", note: str = "") -> ProductIn:
    return ProductIn(title=title, price=price, cost=cost, source=source, note=note)


#: 1688 侧（有采购价）
SUPPLIES = [
    make_product("304不锈钢保温杯 500ml 便携水杯", price=55.5, cost=18.5,
                 source="1688导出"),
    make_product("316L不锈钢保温杯 500ml 高档礼盒", price=120.0, cost=40.0,
                 source="1688导出"),
    make_product("儿童保温杯 吸管杯 学生上学专用水壶", price=45.0, cost=15.0,
                 source="1688导出"),
]

#: 淘宝侧（有零售价、无成本）
TARGETS = [
    make_product("304不锈钢迷你小巧150ml新款保温杯便携男女高颜值", price=9.47),
    make_product("儿童保温杯女生双饮水杯 吸管杯子 学生上学专用水壶", price=59.0),
]


# --------------------------------------------------------------------------- #
# 文本归一化
# --------------------------------------------------------------------------- #

def test_normalize_strips_noise_words_that_create_false_similarity():
    """营销词在两边标题里都大量出现，不剔除会凭空制造相似。"""
    left = normalize_title("新款包邮304不锈钢保温杯500ml正品")
    right = normalize_title("304不锈钢保温杯500ml厂家直销爆款")
    assert left == right == "304不锈钢保温杯500ml"


def test_normalize_strips_highlight_and_punctuation():
    assert normalize_title("<span class=H>保温</span>杯　500ml") == "保温杯500ml"
    assert normalize_title("A-B_C/D") == "abcd"


def test_normalize_is_idempotent():
    once = normalize_title("新款包邮保温杯 500ml")
    assert normalize_title(once) == once


def test_bigrams_basic_and_edge_cases():
    assert bigrams("abc") == {"ab", "bc"}
    assert bigrams("a") == {"a"}
    assert bigrams("") == set()


def test_spec_tokens_extracts_numbers_with_units():
    assert spec_tokens("304不锈钢保温杯500ml") == {"304", "500ml"}
    assert spec_tokens("316L 高档") == {"316l"}
    assert spec_tokens("无规格描述") == set()


def test_shared_specs_sorted():
    assert shared_specs("304不锈钢500ml", "304不锈钢316L") == ["304"]


def test_dice_and_jaccard_bounds():
    assert dice(set(), {"a"}) == 0.0
    assert dice({"a"}, {"a"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


# --------------------------------------------------------------------------- #
# 相似度：真实标题上的行为
# --------------------------------------------------------------------------- #

def test_similarity_same_family_scores_above_unrelated():
    same = title_similarity("304不锈钢保温杯 500ml 便携水杯",
                            "304不锈钢迷你小巧150ml新款保温杯便携")
    unrelated = title_similarity("304不锈钢保温杯 500ml 便携水杯",
                                 "儿童保温杯 吸管杯 学生上学专用")
    assert same > unrelated
    assert unrelated < 0.3


def test_similarity_penalises_material_difference():
    """304 vs 316L 是关键差异，不能被长标题淹没。"""
    same_material = title_similarity("304不锈钢保温杯 500ml", "304不锈钢保温杯 500ml 礼盒")
    diff_material = title_similarity("304不锈钢保温杯 500ml", "316L不锈钢保温杯 500ml 礼盒")
    assert same_material > diff_material
    assert diff_material < same_material


def test_similarity_is_symmetric():
    a, b = "304不锈钢保温杯 500ml", "保温杯 500ml 304不锈钢 便携"
    assert title_similarity(a, b) == title_similarity(b, a)


def test_similarity_identical_is_one():
    assert title_similarity("保温杯 500ml", "保温杯 500ml") == 1.0


def test_similarity_handles_empty_and_none():
    assert title_similarity("", "保温杯") == 0.0
    assert title_similarity(None, None) == 0.0


def test_similarity_ignores_noise_word_differences():
    a = "304不锈钢保温杯500ml"
    b = "新款包邮304不锈钢保温杯500ml正品爆款"
    assert title_similarity(a, b) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 挑供货来源
# --------------------------------------------------------------------------- #

def test_pick_supply_returns_best_above_floor():
    target = TARGETS[0]
    supply, score = pick_supply(target, SUPPLIES, threshold=DEFAULT_THRESHOLD)
    assert supply is not None
    assert supply.cost == 18.5
    assert score > 0


def test_pick_supply_returns_none_when_all_hopeless():
    target = make_product("完全不相干的商品 螺丝刀套装")
    supply, _ = pick_supply(target, SUPPLIES, threshold=DEFAULT_THRESHOLD)
    assert supply is None


def test_pick_supply_ignores_supplies_without_cost():
    supplies = [make_product("304不锈钢保温杯 500ml", cost=0.0)]
    supply, _ = pick_supply(TARGETS[0], supplies)
    assert supply is None


def test_pick_supply_respects_min_shared_specs():
    target = TARGETS[0]
    _, score_loose = pick_supply(target, SUPPLIES, min_shared_specs=0)
    supply, _ = pick_supply(target, SUPPLIES, min_shared_specs=3)
    assert supply is None, "只共享一个 304，要求 3 个规格就不该匹配"
    assert score_loose > 0


# --------------------------------------------------------------------------- #
# 对齐流程
# --------------------------------------------------------------------------- #

def test_link_costs_classifies_by_threshold():
    result = link_costs(TARGETS + SUPPLIES, threshold=0.5)
    assert len(result.matches) == 2
    assert all(match.confidence in {"high", "low", "none"} for match in result.matches)
    assert result.dry_run is True


def test_link_costs_separates_high_and_low_confidence():
    # 阈值拉高到 0.9，原本高置信的会掉到低置信
    strict = link_costs(TARGETS + SUPPLIES, threshold=0.9)
    assert strict.accepted == []
    assert strict.low_confidence

    loose = link_costs(TARGETS + SUPPLIES, threshold=0.2)
    assert loose.accepted
    assert loose.low_confidence == []


def test_link_costs_does_not_touch_products_with_cost():
    """已有成本更可信，不能被覆盖。"""
    products = [make_product("304不锈钢保温杯 500ml 自营", cost=33.0, source="自营")] + SUPPLIES
    result = link_costs(products)
    assert all(match.target.cost <= 0 for match in result.matches)


def test_link_costs_filters_by_source():
    result = link_costs(TARGETS + SUPPLIES, target_source="1688导出")
    assert result.matches == []
    assert any("没有找到「缺成本」" in w for w in result.warnings)


def test_link_costs_supply_source_filter():
    result = link_costs(TARGETS + SUPPLIES, supply_source="不存在的来源")
    assert any("没有找到「有成本」" in w for w in result.warnings)


def test_link_costs_limit():
    result = link_costs(TARGETS + SUPPLIES, limit=1)
    assert len(result.matches) == 1


def test_link_costs_warns_when_no_supply_at_all():
    result = link_costs(TARGETS)
    assert any("有成本" in w for w in result.warnings)


def test_link_costs_warns_on_negative_margin_likely_wrong_match():
    """批发价 > 零售价通常是匹配错了，必须主动诊断。"""
    target = make_product("304不锈钢迷你小巧150ml保温杯便携", price=9.47)
    supply = make_product("304不锈钢保温杯 500ml 便携水杯", cost=18.5, source="1688")
    result = link_costs([target, supply], threshold=0.3)

    assert result.accepted, "阈值 0.3 下应当匹配上"
    assert result.accepted[0].margin < 0
    assert any("匹配错了" in w for w in result.warnings)


def test_link_costs_no_negative_warning_when_margins_healthy():
    target = make_product("304不锈钢保温杯 500ml 便携水杯 正品", price=100.0)
    supply = make_product("304不锈钢保温杯 500ml 便携水杯", cost=20.0, source="1688")
    result = link_costs([target, supply], threshold=0.3)
    assert result.accepted
    assert not any("匹配错了" in w for w in result.warnings)


def test_link_summary_reports_margin_distribution():
    target = make_product("304不锈钢保温杯 500ml 便携水杯 正品", price=100.0)
    supply = make_product("304不锈钢保温杯 500ml 便携水杯", cost=20.0, source="1688")
    summary = link_costs([target, supply], threshold=0.3).summary()
    assert "高置信匹配 1 个" in summary
    assert "毛利率" in summary and "80.0%" in summary


def test_cost_match_margin_zero_when_price_missing():
    match = CostMatch(target=make_product("x", price=0.0), supply=None, score=0.0, cost=10.0)
    assert match.margin == 0.0


# --------------------------------------------------------------------------- #
# 写回
# --------------------------------------------------------------------------- #

def test_with_cost_updates_cost_and_note():
    target = make_product("淘宝杯", price=100.0, note="原始备注")
    supply = make_product("1688杯", cost=25.0, source="1688")
    match = CostMatch(target=target, supply=supply, score=0.62, cost=25.0,
                      shared_specs=["304"])
    updated = with_cost(match)

    assert updated.cost == 25.0
    assert "原始备注" in updated.note
    assert "采购价 25.00 元来自" in updated.note
    assert "0.620" in updated.note
    assert "304" in updated.note
    assert target.cost == 0.0, "不应就地修改原对象"


def test_cost_note_is_idempotent():
    """重复对齐不应叠加多行成本来源。"""
    supply = make_product("1688杯", cost=25.0, source="1688")
    match = CostMatch(target=make_product("淘宝杯"), supply=supply, score=0.6, cost=25.0)
    once = cost_note("备注", match)
    twice = cost_note(once, match)
    assert once == twice
    assert once.count(COST_SEP) == 1


def test_cost_note_marks_low_confidence():
    supply = make_product("1688杯", cost=25.0, source="1688")
    match = CostMatch(target=make_product("淘宝杯"), supply=supply, score=0.42,
                      cost=25.0, confidence="low")
    assert "低置信" in cost_note("", match)
    assert "需人工复核" in cost_note("", match)


def test_cost_note_noop_without_supply():
    match = CostMatch(target=make_product("淘宝杯"), supply=None, score=0.0, cost=0.0)
    assert cost_note("备注", match) == "备注"


def test_apply_matches_writes_to_database(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    for product in TARGETS[:1] + SUPPLIES[:1]:
        db.upsert_product(product, path)

    result = link_costs(db.list_products(db_path=path), threshold=0.4)
    written = apply_matches(result, db_path=path)

    assert written >= 1
    stored = {p.title: p for p in db.list_products(db_path=path)}
    target = stored[TARGETS[0].title]
    assert target.cost > 0
    assert "采购价" in target.note
    assert result.dry_run is False


def test_apply_matches_skips_low_confidence_by_default(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    target = make_product("304不锈钢迷你小巧150ml保温杯便携", price=9.47)
    supply = make_product("304不锈钢保温杯 500ml 便携水杯", cost=18.5, source="1688")
    for product in (target, supply):
        db.upsert_product(product, path)

    # 按实际分数把阈值抬到分数之上，确保落在低置信区间
    score = title_similarity(target.title, supply.title)
    result = link_costs(db.list_products(db_path=path), threshold=min(0.99, score + 0.1))
    assert result.low_confidence, f"相似度 {score:.3f} 应落入低置信区间"
    assert result.accepted == []

    assert apply_matches(result, db_path=path) == 0
    assert {p.title: p for p in db.list_products(db_path=path)}[target.title].cost == 0.0

    assert apply_matches(result, include_low_confidence=True, db_path=path) == 1


def test_apply_matches_ignores_unmatched(tmp_path):
    path = tmp_path / "t.db"
    db.init_db(path)
    result = link_costs(TARGETS)
    assert apply_matches(result, db_path=path) == 0


# --------------------------------------------------------------------------- #
# 毛利率体检
# --------------------------------------------------------------------------- #

def test_margin_report_distribution():
    products = [
        make_product("a", price=100, cost=20),   # 80%
        make_product("b", price=100, cost=50),   # 50%
        make_product("c", price=100, cost=100),  # 0%
        make_product("d", price=100, cost=0),    # 缺成本
    ]
    report = margin_report(products)
    assert report["count"] == 3
    assert report["missing_cost"] == 1
    assert report["median"] == pytest.approx(0.5)
    assert report["min"] == pytest.approx(0.0)
    assert report["max"] == pytest.approx(0.8)


def test_margin_report_flags_saturated_margins():
    """毛利率 ≥95% 基本等于没成本，是数据质量问题，要能数出来。"""
    products = [make_product("a", price=100, cost=0.5), make_product("b", price=100, cost=30)]
    assert margin_report(products)["saturated"] == 1


def test_margin_report_empty():
    assert margin_report([])["count"] == 0
    assert margin_report([make_product("a", price=0, cost=10)])["count"] == 0
