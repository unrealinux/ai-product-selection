"""主图复核（成本对齐的第二个信号）测试。

核心不变量：**算不出相似度时保持原样**。
「缺图 / 下载失败」必须与「图不相似」严格区分 —— 否则缺图的商品会被静默判成不同款，
而这正是这个项目一直在防的那类静默错误。
"""

from __future__ import annotations

from app.costlink import (
    IMAGE_STRONG,
    IMAGE_WEAK,
    CostMatch,
    LinkResult,
    apply_image_signals,
    apply_matches,
    cost_note,
    link_costs,
)
from app.models import ProductIn


def target(title: str = "304不锈钢保温杯 500ml 便携", *, image: str = "https://x.com/a.jpg") -> ProductIn:
    return ProductIn(title=title, source="taobao", price=99.0, cost=0.0, image_url=image)


def supply(title: str = "316L保温杯 500ml", *, cost: float = 22.0,
           image: str = "https://x.com/b.jpg") -> ProductIn:
    return ProductIn(title=title, source="1688", price=50.0, cost=cost, image_url=image)


def make_match(*, score: float = 0.45,
               target_image: str = "https://x.com/a.jpg",
               supply_image: str = "https://x.com/b.jpg") -> CostMatch:
    item = supply(image=supply_image)
    return CostMatch(target=target(image=target_image), supply=item, score=score,
                     cost=item.cost, confidence="low", shared_specs=["500ml"])


def make_result(*matches: CostMatch) -> LinkResult:
    return LinkResult(matches=list(matches))


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #

def test_high_image_similarity_promotes_to_high_confidence():
    match = make_match()
    report = apply_image_signals(make_result(match), similarity_fn=lambda a, b: 0.95)

    assert report.promoted == 1
    assert report.compared == 1
    assert match.confidence == "high"
    assert match.method == "image"
    assert match.image_score == 0.95


def test_low_image_similarity_marks_rejected():
    match = make_match()
    report = apply_image_signals(make_result(match), similarity_fn=lambda a, b: 0.10)

    assert report.rejected == 1
    assert match.confidence == "rejected"
    assert match.image_score == 0.10


def test_middle_image_similarity_stays_low_confidence():
    match = make_match()
    report = apply_image_signals(make_result(match), similarity_fn=lambda a, b: 0.55)

    assert report.ambiguous == 1
    assert match.confidence == "low"
    assert match.image_score == 0.55  # 仍然记录，供人看


def test_thresholds_are_inclusive_boundaries():
    strong_match = make_match()
    apply_image_signals(make_result(strong_match), similarity_fn=lambda a, b: IMAGE_STRONG)
    assert strong_match.confidence == "high"

    weak_match = make_match()
    apply_image_signals(make_result(weak_match), similarity_fn=lambda a, b: IMAGE_WEAK)
    assert weak_match.confidence == "rejected"


# --------------------------------------------------------------------------- #
# 缺图与失败必须原样保留
# --------------------------------------------------------------------------- #

def test_missing_target_image_keeps_candidate_untouched():
    match = make_match(target_image="")
    report = apply_image_signals(make_result(match), similarity_fn=lambda a, b: 0.99)

    assert report.no_image == 1
    assert report.compared == 0
    assert match.confidence == "low"
    assert match.image_score is None


def test_missing_supply_image_keeps_candidate_untouched():
    match = make_match(supply_image="")
    report = apply_image_signals(make_result(match), similarity_fn=lambda a, b: 0.01)

    assert report.no_image == 1
    assert match.confidence == "low"


def test_uncomputable_similarity_is_not_treated_as_dissimilar():
    """返回 None（下载失败/解码失败）时绝不能判为不同款。"""
    match = make_match()
    report = apply_image_signals(make_result(match), similarity_fn=lambda a, b: None)

    assert report.failed == 1
    assert report.rejected == 0
    assert match.confidence == "low"
    assert match.image_score is None


def test_no_candidates_reports_cleanly():
    report = apply_image_signals(make_result())
    assert report.compared == 0
    assert report.notes


def test_notes_explain_skipped_missing_images():
    result = make_result(make_match(target_image=""), make_match(supply_image=""))
    report = apply_image_signals(result, similarity_fn=lambda a, b: 0.9)
    assert report.no_image == 2
    assert any("没有主图" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# 与其他环节的衔接
# --------------------------------------------------------------------------- #

def test_image_rejected_is_never_written(tmp_path):
    from app import db

    match = make_match()
    result = make_result(match)
    apply_image_signals(result, similarity_fn=lambda a, b: 0.05)

    db_path = tmp_path / "x.db"
    db.init_db(db_path)
    assert apply_matches(result, include_low_confidence=True, db_path=db_path) == 0


def test_image_promoted_is_written_with_auditable_note(tmp_path):
    from app import db

    match = make_match()
    result = make_result(match)
    apply_image_signals(result, similarity_fn=lambda a, b: 0.93)

    db_path = tmp_path / "x.db"
    db.init_db(db_path)
    assert apply_matches(result, db_path=db_path) == 1

    note = cost_note("", match)
    assert "低置信经主图复核通过" in note
    assert "主图相似度 0.930" in note
    assert "需人工复核" in note


def test_image_promoted_candidates_are_not_re_reviewed_by_llm():
    """主图已经升级为高置信的，不该再进大模型的灰区队列。"""
    from app.match_review import review_matches

    match = make_match()
    result = make_result(match)
    apply_image_signals(result, similarity_fn=lambda a, b: 0.95)

    report = review_matches(result)  # 没有候选时应直接返回，不调用模型
    assert report.reviewed == 0
    assert result.low_confidence == []


# --------------------------------------------------------------------------- #
# 与 link_costs 的端到端衔接
# --------------------------------------------------------------------------- #

def test_link_costs_then_image_review_end_to_end():
    products = [target(), supply()]
    result = link_costs(products)
    assert result.low_confidence, "这条应落在灰区（相似度约 0.47）"

    report = apply_image_signals(result, similarity_fn=lambda a, b: 0.91)

    assert report.promoted == 1
    promoted = result.accepted[0]
    assert promoted.method == "image"
    assert promoted.target.image_url and promoted.supply.image_url
