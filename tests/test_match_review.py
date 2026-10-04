"""低置信匹配的大模型复核测试。

盯住四件容易出错的事：

1. **宁可不判**：模型置信度不足时必须保持低置信，不能升级也不能删除
2. **编号串位**：模型返回的 index 必须在本次请求的候选集合内，否则丢弃
3. **布尔与数值的脏输入**：``"false"`` / ``0`` / ``200`` 这类都要能正确解析
4. **缓存与降级**：重复运行不重复花 token；没有 LLM 时原样返回、不静默改数据
"""

from __future__ import annotations

import json

import pytest

from app import llm
from app.costlink import CostMatch, LinkResult, cost_note, link_costs
from app.match_review import (
    ReviewCache,
    ReviewVerdict,
    apply_verdicts,
    build_pair_prompt,
    parse_verdicts,
    review_matches,
)
from app.models import ProductIn


# --------------------------------------------------------------------------- #
# 造数据
# --------------------------------------------------------------------------- #

def target(title: str = "304不锈钢保温杯 500ml", price: float = 100.0) -> ProductIn:
    return ProductIn(title=title, source="taobao", price=price, cost=0.0)


def supply(title: str = "316L保温杯 500ml", cost: float = 40.0) -> ProductIn:
    return ProductIn(title=title, source="1688", price=0.0, cost=cost)


def make_match(target_title: str = "304不锈钢保温杯 500ml",
               supply_title: str = "316L保温杯 500ml", *, score: float = 0.45,
               confidence: str = "low", specs: list[str] | None = None) -> CostMatch:
    item = supply(supply_title)
    return CostMatch(target=target(target_title), supply=item, score=score,
                     cost=item.cost, confidence=confidence,
                     shared_specs=specs if specs is not None else ["500ml"])


def make_result(*matches: CostMatch) -> LinkResult:
    return LinkResult(matches=list(matches))


@pytest.fixture(autouse=True)
def _llm_available(monkeypatch):
    """默认把 LLM 视为可用，逐条测试里再覆盖具体行为。"""
    monkeypatch.setattr(llm, "is_available", lambda: True)


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #

def test_prompt_contains_everything_the_model_needs():
    match = make_match()
    prompt = build_pair_prompt([(3, match)])

    assert "第 3 组" in prompt
    assert match.target.title in prompt
    assert match.supply_title in prompt
    assert "0.450" in prompt
    assert "500ml" in prompt
    assert "100.00" in prompt and "40.00" in prompt


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #

def test_parse_verdicts_from_dict_and_list():
    payload = {"items": [
        {"index": 0, "same_product": True, "confidence": 90, "reason": "规格一致"},
        {"index": 1, "same_product": False, "confidence": 80, "reason": "材质不同"},
    ]}
    assert parse_verdicts(payload, 2)[0].same_product is True
    assert parse_verdicts(payload["items"], 2)[1].same_product is False


def test_parse_verdicts_rejects_bad_payload():
    assert parse_verdicts(None, 2) == {}
    assert parse_verdicts("不是 JSON", 2) == {}
    assert parse_verdicts({"items": "x"}, 2) == {}


def test_parse_verdicts_coerces_dirty_values():
    payload = {"items": [
        {"index": "0", "same_product": "false", "confidence": 200, "reason": "x"},
        {"index": 1, "same_product": 0, "confidence": -5, "reason": "y"},
        {"index": 2, "same_product": "是", "confidence": "85.6", "reason": "z"},
    ]}
    parsed = parse_verdicts(payload, 3)

    assert parsed[0].same_product is False
    assert parsed[0].confidence == 100  # 截断到 100
    assert parsed[1].same_product is False
    assert parsed[1].confidence == 0
    assert parsed[2].same_product is True
    assert parsed[2].confidence == 85


def test_parse_verdicts_drops_out_of_range_index():
    payload = {"items": [{"index": 5, "same_product": True, "confidence": 90}]}
    assert parse_verdicts(payload, 3, allowed={0, 1, 2}) == {}


def test_parse_verdicts_skips_non_mapping_items():
    payload = {"items": ["垃圾", {"index": 0, "same_product": True, "confidence": 90}]}
    assert set(parse_verdicts(payload, 2)) == {0}


# --------------------------------------------------------------------------- #
# 应用判断
# --------------------------------------------------------------------------- #

def test_apply_verdicts_promotes_confident_same_product():
    matches = [make_match()]
    promoted, rejected, kept = apply_verdicts(
        matches,
        {0: ReviewVerdict(index=0, same_product=True, confidence=90, reason="规格一致")},
        min_confidence=70,
    )
    assert (promoted, rejected, kept) == (1, 0, 0)
    assert matches[0].confidence == "high"
    assert matches[0].method == "llm_review"
    assert matches[0].review_reason == "规格一致"
    assert matches[0].review_confidence == 90


def test_apply_verdicts_rejects_confident_different_product():
    matches = [make_match()]
    promoted, rejected, kept = apply_verdicts(
        matches,
        {0: ReviewVerdict(index=0, same_product=False, confidence=88, reason="材质不同")},
        min_confidence=70,
    )
    assert (promoted, rejected, kept) == (0, 1, 0)
    assert matches[0].confidence == "rejected"


def test_apply_verdicts_keeps_low_when_not_confident():
    """置信度不足时必须保持低置信 —— 这是「宁可不判」的核心。"""
    matches = [make_match()]
    promoted, rejected, kept = apply_verdicts(
        matches,
        {0: ReviewVerdict(index=0, same_product=True, confidence=55, reason="可能吧")},
        min_confidence=70,
    )
    assert (promoted, rejected, kept) == (0, 0, 1)
    assert matches[0].confidence == "low"
    # 理由仍然记录下来，便于人工判断
    assert matches[0].review_reason == "可能吧"


def test_apply_verdicts_keeps_low_when_model_is_unsure():
    matches = [make_match()]
    promoted, rejected, kept = apply_verdicts(
        matches, {0: ReviewVerdict(index=0, same_product=None, confidence=99)},
        min_confidence=70,
    )
    assert (promoted, rejected, kept) == (0, 0, 1)
    assert matches[0].confidence == "low"


def test_apply_verdicts_ignores_out_of_range_and_missing_supply():
    matches = [make_match(), CostMatch(target=target(), supply=None, score=0.2, cost=0.0,
                                       confidence="none")]
    promoted, rejected, kept = apply_verdicts(
        matches,
        {
            5: ReviewVerdict(index=5, same_product=True, confidence=90),
            1: ReviewVerdict(index=1, same_product=True, confidence=90),
        },
        min_confidence=70,
    )
    assert (promoted, rejected, kept) == (0, 0, 0)


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #

def test_review_cache_round_trip(tmp_path):
    path = tmp_path / "reviews.json"
    cache = ReviewCache(path)
    cache.put("标题A", "标题B", {"same_product": True, "confidence": 90, "reason": "同款"})
    cache.save()

    reloaded = ReviewCache(path)
    assert reloaded.get("标题A", "标题B")["confidence"] == 90
    # 归一化后等价（去噪词 / 标点）也应命中同一键
    assert reloaded.get("标题 A ", "标题B！") is not None


def test_review_cache_disabled_is_noop(tmp_path):
    cache = ReviewCache(tmp_path / "x.json", enabled=False)
    cache.put("a", "b", {"same_product": True})
    cache.save()
    assert cache.get("a", "b") is None
    assert not (tmp_path / "x.json").exists()


def test_review_cache_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ 不是合法 JSON", encoding="utf-8")
    cache = ReviewCache(path)  # 不应抛异常
    assert cache.get("a", "b") is None


# --------------------------------------------------------------------------- #
# 复核流程
# --------------------------------------------------------------------------- #

def payload_for(index: int, same: bool, confidence: int, reason: str = "理由") -> str:
    return json.dumps({"items": [
        {"index": index, "same_product": same, "confidence": confidence, "reason": reason}
    ]}, ensure_ascii=False)


def test_review_without_candidates_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("不应调用模型"))
    report = review_matches(make_result(), cache=ReviewCache(tmp_path / "c.json"))
    assert report.reviewed == 0
    assert report.notes


def test_review_degrades_without_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda: False)
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("不应调用模型"))
    result = make_result(make_match())

    report = review_matches(result, cache=ReviewCache(tmp_path / "c.json"))

    assert report.reviewed == 0
    assert result.matches[0].confidence == "low"  # 原样保留
    assert any("未配置" in note for note in report.notes)


def test_review_promotes_and_rejects(tmp_path, monkeypatch):
    good, bad = make_match(), make_match("儿童保温杯", "316L保温杯")
    result = make_result(good, bad)

    def fake_chat(messages, temperature=0.1):
        return json.dumps({"items": [
            {"index": 0, "same_product": True, "confidence": 92, "reason": "同规格"},
            {"index": 1, "same_product": False, "confidence": 88, "reason": "人群不同"},
        ]}, ensure_ascii=False)

    monkeypatch.setattr(llm, "chat", fake_chat)
    report = review_matches(result, cache=ReviewCache(tmp_path / "c.json"))

    assert (report.promoted, report.rejected) == (1, 1)
    assert good.confidence == "high" and good.method == "llm_review"
    assert bad.confidence == "rejected"
    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert result.low_confidence == []


def test_review_uses_cache_on_second_run(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_chat(messages, temperature=0.1):
        calls["n"] += 1
        return payload_for(0, True, 90)

    monkeypatch.setattr(llm, "chat", fake_chat)
    cache_path = tmp_path / "c.json"

    first = make_result(make_match())
    review_matches(first, cache=ReviewCache(cache_path))
    assert first.matches[0].confidence == "high"

    second = make_result(make_match())
    report = review_matches(second, cache=ReviewCache(cache_path))

    assert calls["n"] == 1  # 第二次全部命中缓存
    assert report.cached == 1
    assert second.matches[0].confidence == "high"


def test_review_records_errors_without_breaking(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: None)
    result = make_result(make_match())
    report = review_matches(result, cache=ReviewCache(tmp_path / "c.json"))
    assert report.errors
    assert result.matches[0].confidence == "low"

    monkeypatch.setattr(llm, "chat", lambda *a, **k: "模型胡说八道")
    result = make_result(make_match())
    report = review_matches(result, cache=ReviewCache(tmp_path / "d.json"))
    assert report.errors


def test_review_respects_min_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: payload_for(0, True, 60))
    result = make_result(make_match())

    report = review_matches(result, min_confidence=80, cache=ReviewCache(tmp_path / "c.json"))

    assert report.promoted == 0
    assert report.kept == 1
    assert result.matches[0].confidence == "low"


def test_review_does_not_touch_high_confidence_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: payload_for(0, False, 99))
    high = make_match(confidence="high", score=0.8)
    low = make_match()
    result = make_result(high, low)

    review_matches(result, cache=ReviewCache(tmp_path / "c.json"))

    assert high.confidence == "high"  # 只复核灰区
    assert high.review_reason == ""
    assert low.confidence == "rejected"


# --------------------------------------------------------------------------- #
# 与成本对齐的衔接
# --------------------------------------------------------------------------- #

def test_promoted_match_is_written_and_note_is_auditable(tmp_path, monkeypatch):
    products = [
        ProductIn(title="304不锈钢保温杯 500ml 便携", source="taobao", price=100.0, cost=0.0),
        ProductIn(title="316L保温杯 500ml", source="1688", price=0.0, cost=40.0),
    ]
    result = link_costs(products)  # 默认阈值 0.5 → 0.4667 落在灰区
    assert result.low_confidence, "这条应落入低置信区间"

    monkeypatch.setattr(llm, "chat", lambda *a, **k: payload_for(0, True, 91, "同规格同形态"))
    report = review_matches(result, cache=ReviewCache(tmp_path / "c.json"))

    assert report.promoted >= 1
    promoted = next(m for m in result.accepted if m.method == "llm_review")
    note = cost_note("", promoted)
    assert "低置信经大模型复核通过" in note
    assert "大模型复核：同规格同形态（置信 91）" in note
    assert "需人工复核" in note


def test_rejected_match_is_not_written(tmp_path, monkeypatch):
    from app import db
    from app.costlink import apply_matches

    products = [
        ProductIn(title="304不锈钢保温杯 500ml 便携", source="taobao", price=100.0, cost=0.0),
        ProductIn(title="316L保温杯 500ml", source="1688", price=0.0, cost=40.0),
    ]
    result = link_costs(products)
    assert result.low_confidence
    monkeypatch.setattr(llm, "chat", lambda *a, **k: payload_for(0, False, 95, "材质不同"))
    review_matches(result, cache=ReviewCache(tmp_path / "c.json"))

    db_path = tmp_path / "x.db"
    db.init_db(db_path)
    # 即使加了 include_low_confidence，判定非同款的也不该写入
    assert apply_matches(result, include_low_confidence=True, db_path=db_path) == 0
