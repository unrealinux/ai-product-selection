"""维度补齐测试：商品详情接口取重量 + 大模型估算重量/复购/合规。

重量解析的字段与单位定义来自官方文档 https://op.jinritemai.com/docs/api-docs/14/56
（``weight_value`` / ``weight_unit``：0=kg，1=g；SKU 级 ``delivery_infos`` 支持 mg/g/kg）。
"""

from __future__ import annotations

import json

import pytest

from app import enrich as enrich_mod
from app.enrich import (
    PROVENANCE_SEP,
    EstimateCache,
    EnrichReport,
    describe_sources,
    enrich,
    estimate_with_llm,
    fill_weight_from_detail,
    note_body,
    with_sources,
)
from app.models import ProductIn
from app.sources.douyin import (
    DETAIL_UNIT_TO_KG,
    WEIGHT_UNIT_TO_KG,
    DouyinAPIError,
    DouyinClient,
    parse_weight_kg,
)

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #

def make_product(**overrides) -> ProductIn:
    base = dict(
        title="冷萃冻干咖啡粉 30 条装",
        external_id="3400312511185213726",
        category="抖音类目-2634",
        price=20.0,
        cost=14.0,
        source="douyin",
        heat=79.4,
        competition=22.0,
        weight_kg=0.5,
        repurchase=50.0,
        compliance_risk=20.0,
        virality=100.0,
        note="抖音精选联盟｜商品ID 3400312511185213726 店铺 示范小铺｜数据说明：热度/竞争度/"
             "毛利率/传播力由接口字段推导；重量、复购、合规为缺省值，需人工复核",
    )
    base.update(overrides)
    return ProductIn(**base)


class FakeDetailClient:
    """替身：按 product_id 返回预设详情，或抛错。"""

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get_product_detail(self, product_id=None, out_product_id=None, show_draft=False):
        self.calls.append(str(product_id))
        result = self.responses.get(str(product_id))
        if isinstance(result, Exception):
            raise result
        return result or {}


@pytest.fixture
def llm_available(monkeypatch):
    monkeypatch.setattr(enrich_mod.llm, "is_available", lambda: True)


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(enrich_mod.llm, "is_available", lambda: False)


# --------------------------------------------------------------------------- #
# 重量解析
# --------------------------------------------------------------------------- #

def test_weight_unit_table_matches_official_doc():
    """官方：weight_unit 0-kg, 1-g；SKU 级支持 mg/g/kg。"""
    assert WEIGHT_UNIT_TO_KG == {0: 1.0, 1: 0.001}
    assert DETAIL_UNIT_TO_KG["kg"] == 1.0
    assert DETAIL_UNIT_TO_KG["g"] == 0.001
    assert DETAIL_UNIT_TO_KG["mg"] == 0.000001


@pytest.mark.parametrize(
    "detail,expected",
    [
        ({"weight_value": "0.35", "weight_unit": "0"}, 0.35),      # 0 = kg
        ({"weight_value": "350", "weight_unit": "1"}, 0.35),       # 1 = g
        ({"weight_value": 2.4, "weight_unit": 0}, 2.4),            # 数字类型
        ({"weight_value": "0", "weight_unit": "0"}, None),         # 0 视为未设置
        ({"weight_value": "", "weight_unit": "0"}, None),
        ({}, None),
        ({"weight_value": "abc"}, None),
    ],
)
def test_parse_weight_kg_product_level(detail, expected):
    assert parse_weight_kg(detail) == expected


def test_parse_weight_kg_falls_back_to_sku_level_and_takes_max():
    """商品级为 0 时回落到 SKU 级；多规格取最大重量（发货按最重规格估算更安全）。"""
    detail = {
        "weight_value": "0",
        "spec_prices": [
            {"delivery_infos": [
                {"info_type": "weight", "info_value": "800", "info_unit": "g"},
                {"info_type": "weight", "info_value": "1", "info_unit": "kg"},
            ]},
            {"delivery_infos": [
                {"info_type": "weight", "info_value": "500000", "info_unit": "mg"},
            ]},
        ],
    }
    assert parse_weight_kg(detail) == pytest.approx(1.0)


def test_parse_weight_kg_ignores_non_weight_and_unknown_units():
    detail = {"spec_prices": [{"delivery_infos": [
        {"info_type": "volume", "info_value": "10", "info_unit": "g"},
        {"info_type": "weight", "info_value": "10", "info_unit": "斤"},
        {"info_type": "weight", "info_value": "bad", "info_unit": "g"},
    ]}]}
    assert parse_weight_kg(detail) is None


def test_parse_weight_kg_does_not_use_cross_border_net_weight():
    """logistics_info.net_weight_qty 文档未标单位，不采用。"""
    assert parse_weight_kg({"logistics_info": {"net_weight_qty": "100"}}) is None


def test_parse_weight_kg_unknown_unit_falls_back_to_kg():
    """单位缺失/非法时按 kg 处理，而不是丢弃数值。"""
    assert parse_weight_kg({"weight_value": "1.5", "weight_unit": ""}) == 1.5
    assert parse_weight_kg({"weight_value": "1.5", "weight_unit": "9"}) == 1.5


# --------------------------------------------------------------------------- #
# 商品详情接口调用
# --------------------------------------------------------------------------- #

def test_get_product_detail_builds_official_request():
    from tests.test_douyin import FakeResponse, FakeSession

    session = FakeSession(FakeResponse({"code": 10000, "data": {"weight_value": "0.4",
                                                               "weight_unit": "0"}}))
    client = DouyinClient("k" * 19, "secret", access_token="t", session=session)
    detail = client.get_product_detail("3400312511185213726")

    call = session.calls[0]
    assert call["url"].endswith("/product/detail")
    assert call["params"]["method"] == "product.detail"
    # show_draft 不传时不下发（官方默认 false）
    assert json.loads(call["data"]) == {"product_id": "3400312511185213726"}
    assert parse_weight_kg(detail) == 0.4


def test_get_product_detail_sends_show_draft_only_when_true():
    from tests.test_douyin import FakeResponse, FakeSession

    session = FakeSession(FakeResponse({"code": 10000, "data": {}}))
    client = DouyinClient("k" * 19, "secret", access_token="t", session=session)
    client.get_product_detail(out_product_id="SKU-1", show_draft=True)
    assert json.loads(session.calls[0]["data"]) == {
        "out_product_id": "SKU-1", "show_draft": "true"
    }


def test_get_product_detail_requires_identifier():
    from tests.test_douyin import FakeSession

    client = DouyinClient("k" * 19, "secret", access_token="t", session=FakeSession())
    with pytest.raises(ValueError, match="product_id"):
        client.get_product_detail()


# --------------------------------------------------------------------------- #
# 用详情接口补重量
# --------------------------------------------------------------------------- #

def test_fill_weight_from_detail_success():
    product = make_product()
    client = FakeDetailClient({"3400312511185213726": {"weight_value": "1.2", "weight_unit": "0"}})
    report = EnrichReport()
    sources: dict[int, dict[str, str]] = {}

    result = fill_weight_from_detail([product], client, report=report, sources=sources)

    assert result[0].weight_kg == 1.2
    assert sources[0]["weight_kg"] == "商品详情接口"
    assert report.detail_attempted == 1 and report.detail_filled == 1


def test_fill_weight_from_detail_handles_not_found_for_other_shops_products():
    """精选联盟商品属于别人店铺，详情接口会报「商品不存在」，不能中断流程。"""
    product = make_product()
    error = DouyinAPIError(40004, "非法的参数", "isv.parameter-invalid:2010058", "商品不存在")
    client = FakeDetailClient({"3400312511185213726": error})
    report = EnrichReport()

    result = fill_weight_from_detail([product], client, report=report)

    assert result[0].weight_kg == 0.5  # 保持缺省值
    assert report.detail_failed == 1 and report.detail_filled == 0
    assert any("只能查已授权店铺自己的商品" in note for note in report.notes)
    assert report.errors and "商品不存在" in report.errors[0]


def test_fill_weight_from_detail_respects_limit():
    product = make_product()
    client = FakeDetailClient({})
    report = EnrichReport()
    fill_weight_from_detail([product] * 5, client, limit=2, report=report)
    assert report.detail_attempted == 2


def test_fill_weight_from_detail_skips_products_without_external_id():
    product = make_product(external_id="", note="手工录入，没有商品ID")
    report = EnrichReport()
    fill_weight_from_detail([product], FakeDetailClient({}), report=report)
    assert report.detail_attempted == 0


def test_fill_weight_from_detail_falls_back_to_note_for_legacy_rows():
    """external_id 字段引入前导入的老数据，仍能从 note 里取回商品 ID。"""
    product = make_product(external_id="")
    client = FakeDetailClient({"3400312511185213726": {"weight_value": "0.9", "weight_unit": "0"}})
    result = fill_weight_from_detail([product], client)
    assert client.calls == ["3400312511185213726"]
    assert result[0].weight_kg == 0.9


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #

def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    product = make_product()
    cache = EstimateCache(path)
    assert cache.get(product) is None

    cache.put(product, {"index": 0, "weight_kg": 0.4, "repurchase": 60})
    cache.save()

    reloaded = EstimateCache(path)
    assert reloaded.get(product)["weight_kg"] == 0.4
    assert len(reloaded) == 1


def test_cache_key_depends_on_title_and_category():
    a = make_product()
    b = make_product(title="另一种商品")
    assert EstimateCache.make_key(a) == EstimateCache.make_key(make_product())
    assert EstimateCache.make_key(a) != EstimateCache.make_key(b)


def test_cache_disabled_writes_nothing(tmp_path):
    path = tmp_path / "cache.json"
    cache = EstimateCache(path, enabled=False)
    cache.put(make_product(), {"weight_kg": 1.0})
    cache.save()
    assert not path.exists()
    assert cache.get(make_product()) is None


def test_cache_survives_corrupted_file(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{ this is not json", encoding="utf-8")
    cache = EstimateCache(path)  # 不应抛错
    assert len(cache) == 0
    assert cache.get(make_product()) is None


# --------------------------------------------------------------------------- #
# 大模型估算
# --------------------------------------------------------------------------- #

def _stub_chat(monkeypatch, payload, recorder=None):
    def fake_chat(messages, temperature=0.3):
        if recorder is not None:
            recorder.append(messages)
        return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    monkeypatch.setattr(enrich_mod.llm, "chat", fake_chat)


def test_estimate_with_llm_applies_values(llm_available, monkeypatch, tmp_path):
    payload = {"items": [
        {"index": 0, "weight_kg": 0.4, "repurchase": 62, "compliance_risk": 14,
         "reason": "食品类复购高"},
        {"index": 1, "weight_kg": 0.35, "repurchase": 22, "compliance_risk": 18,
         "reason": "耐用品"},
    ]}
    _stub_chat(monkeypatch, payload)
    products = [make_product(), make_product(title="便携保温杯 500ml")]
    report = EnrichReport()
    sources: dict[int, dict[str, str]] = {}

    result = estimate_with_llm(
        products, report=report, sources=sources,
        cache=EstimateCache(tmp_path / "c.json"),
    )

    assert result[0].weight_kg == 0.4
    assert result[0].repurchase == 62
    assert result[0].compliance_risk == 14
    assert result[1].weight_kg == 0.35
    assert sources[0]["weight_kg"] == "大模型估算"
    assert report.llm_requested == 2 and report.llm_filled == 2


def test_estimate_with_llm_clamps_out_of_range(llm_available, monkeypatch, tmp_path):
    payload = {"items": [{"index": 0, "weight_kg": -5, "repurchase": 999,
                          "compliance_risk": -20}]}
    _stub_chat(monkeypatch, payload)
    result = estimate_with_llm(
        [make_product()], cache=EstimateCache(tmp_path / "c.json")
    )
    assert result[0].weight_kg == 0.01     # 下限
    assert result[0].repurchase == 100.0   # 上限
    assert result[0].compliance_risk == 0.0


def test_estimate_with_llm_handles_markdown_fence(llm_available, monkeypatch, tmp_path):
    _stub_chat(monkeypatch, '```json\n{"items":[{"index":0,"weight_kg":0.3}]}\n```')
    result = estimate_with_llm([make_product()], cache=EstimateCache(tmp_path / "c.json"))
    assert result[0].weight_kg == 0.3


def test_estimate_with_llm_records_unparsable_response(llm_available, monkeypatch, tmp_path):
    _stub_chat(monkeypatch, "抱歉，我无法判断。")
    report = EnrichReport()
    result = estimate_with_llm(
        [make_product()], report=report, cache=EstimateCache(tmp_path / "c.json")
    )
    assert result[0].weight_kg == 0.5  # 原样返回
    assert report.llm_filled == 0
    assert any("无法解析" in e for e in report.errors)


def test_estimate_with_llm_ignores_out_of_range_index(llm_available, monkeypatch, tmp_path):
    _stub_chat(monkeypatch, {"items": [
        {"index": 7, "weight_kg": 9.9}, {"index": 0, "weight_kg": 0.4}
    ]})
    result = estimate_with_llm([make_product()], cache=EstimateCache(tmp_path / "c.json"))
    assert result[0].weight_kg == 0.4


def test_estimate_with_llm_skips_when_not_configured(no_llm, monkeypatch, tmp_path):
    called = []
    _stub_chat(monkeypatch, {"items": []}, recorder=called)
    report = EnrichReport()
    result = estimate_with_llm(
        [make_product()], report=report, cache=EstimateCache(tmp_path / "c.json")
    )
    assert called == [], "未配置 LLM 时不应发起调用"
    assert result[0].weight_kg == 0.5
    assert any("未配置 APS_LLM_*" in note for note in report.notes)


def test_estimate_with_llm_uses_cache_and_skips_second_call(llm_available, monkeypatch, tmp_path):
    calls = []
    _stub_chat(monkeypatch, {"items": [{"index": 0, "weight_kg": 0.4, "repurchase": 62}]},
               recorder=calls)
    cache = EstimateCache(tmp_path / "c.json")
    first = estimate_with_llm([make_product()], cache=cache)
    assert len(calls) == 1 and first[0].weight_kg == 0.4

    report = EnrichReport()
    second = estimate_with_llm([make_product()], report=report, cache=cache)
    assert len(calls) == 1, "第二次应命中缓存，不再调用大模型"
    assert second[0].weight_kg == 0.4
    assert report.llm_cached == 1 and report.llm_filled == 1


def test_estimate_with_llm_batches_requests(llm_available, monkeypatch, tmp_path):
    calls = []
    _stub_chat(monkeypatch, {"items": [{"index": 0, "weight_kg": 0.4}]}, recorder=calls)
    products = [make_product(title=f"商品{i}") for i in range(5)]
    estimate_with_llm(products, batch_size=2, cache=EstimateCache(tmp_path / "c.json"))
    assert len(calls) == 3  # 5 条按每批 2 条 → 3 批


def test_estimate_prompt_includes_titles(llm_available, monkeypatch, tmp_path):
    calls = []
    _stub_chat(monkeypatch, {"items": [{"index": 0, "weight_kg": 0.4}]}, recorder=calls)
    estimate_with_llm([make_product()], cache=EstimateCache(tmp_path / "c.json"))
    user_prompt = calls[0][1]["content"]
    assert "冷萃冻干咖啡粉 30 条装" in user_prompt
    assert "抖音类目-2634" in user_prompt


# --------------------------------------------------------------------------- #
# note 数据来源说明
# --------------------------------------------------------------------------- #

def test_describe_sources_default_and_custom():
    assert "缺省值" in describe_sources({})
    text = describe_sources({"weight_kg": "大模型估算", "repurchase": "大模型估算"})
    assert "重量=大模型估算" in text and "复购=大模型估算" in text


def test_with_sources_replaces_previous_provenance():
    note = make_product().note
    updated = with_sources(note, {"weight_kg": "商品详情接口"})
    assert updated.count(PROVENANCE_SEP) == 1
    assert "重量=商品详情接口" in updated
    assert "人工复核" not in updated
    assert note_body(updated) == note_body(note)


# --------------------------------------------------------------------------- #
# enrich 编排
# --------------------------------------------------------------------------- #

def test_enrich_reports_and_rewrites_provenance(llm_available, monkeypatch, tmp_path):
    _stub_chat(monkeypatch, {"items": [{"index": 0, "weight_kg": 0.4,
                                        "repurchase": 62, "compliance_risk": 14}]})
    client = FakeDetailClient({"3400312511185213726": {"weight_value": "0.45", "weight_unit": "0"}})

    result, report = enrich(
        [make_product()], client=client, use_llm=True,
        cache=EstimateCache(tmp_path / "c.json"),
    )

    product = result[0]
    # 详情接口的成功值优先保留（大模型只覆盖自己返回的字段）
    assert product.weight_kg == 0.4
    assert product.repurchase == 62 and product.compliance_risk == 14
    assert "重量=大模型估算" in product.note
    assert report.detail_filled == 1 and report.llm_filled == 1
    assert "详情接口" in report.summary() and "大模型估算" in report.summary()


def test_enrich_without_any_source_keeps_original_note(no_llm, tmp_path):
    product = make_product()
    result, report = enrich(
        [product], client=None, use_llm=True, cache=EstimateCache(tmp_path / "c.json")
    )
    assert result[0].note == product.note, "未补齐任何字段时不应改动 note"
    assert result[0].weight_kg == 0.5


def test_enrich_weight_only_from_detail_no_llm(no_llm):
    client = FakeDetailClient({"3400312511185213726": {"weight_value": "1.2", "weight_unit": "0"}})
    result, report = enrich([make_product()], client=client, use_llm=False)
    assert result[0].weight_kg == 1.2
    assert "重量=商品详情接口" in result[0].note
    assert report.llm_requested == 0


def test_enrich_empty_input():
    result, report = enrich([], use_llm=False)
    assert result == [] and report.filled == 0
