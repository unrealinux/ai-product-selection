"""淘宝 A2A 数据源测试。

重点盯住会**静默出错**的地方：

1. 搜索价 vs 真实售价（实测差 78%）—— 用错价格会让毛利率整个错掉
2. 容量被当成重量（``240mL`` vs ``150g``）—— 差着密度，物流维度会歪
3. 广告位未过滤、跨关键词重复、详情批次边界
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.sources.taobao import (
    ARTIFACT_DETAIL,
    ARTIFACT_SEARCH,
    MAX_COMPARE_IDS,
    MAX_DETAIL_IDS,
    RANK_HEAT_BOTTOM,
    RANK_HEAT_TOP,
    SKILL_DETAIL,
    SKILL_SEARCH,
    A2AResult,
    TaobaoA2AClient,
    TaobaoA2AError,
    TaobaoSource,
    TaobaoTaskFailed,
    extract_weight_kg,
    rank_to_heat,
    strip_highlight,
    to_product,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def fixture_search_products() -> list[dict]:
    data = load_fixture("taobao_search.json")
    return data["result"]["task"]["artifacts"][0]["parts"][0]["data"]["data"]["products"]


def fixture_detail_item() -> dict:
    data = load_fixture("taobao_detail.json")
    return data["result"]["task"]["artifacts"][0]["parts"][0]["data"]["data"]["items"][0]


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #

class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str | None = None,
                 raise_on_json: bool = False) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)
        self._raise = raise_on_json

    def json(self):
        if self._raise:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def completed(artifact: str, data: dict) -> dict:
    """构造一个 TASK_STATE_COMPLETED 的 A2A 响应。"""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "result": {
            "task": {
                "id": str(uuid.uuid4()),
                "contextId": str(uuid.uuid4()),
                "status": {"state": "TASK_STATE_COMPLETED", "message": None},
                "artifacts": [{
                    "artifactId": str(uuid.uuid4()),
                    "name": artifact,
                    "parts": [{"data": {"result": "ok", "data": data}}],
                }],
            }
        },
    }


def failed(text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "task": {
                "id": "t1",
                "status": {
                    "state": "TASK_STATE_FAILED",
                    "message": {"role": "ROLE_AGENT", "parts": [{"text": text}]},
                },
                "artifacts": None,
            }
        },
    }


class FakeSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url, timeout=None):
        self.calls.append({"url": url})
        return self.responses.pop(0)


class FakeClient:
    """替身 A2A 客户端：按 itemId 返回预设详情。"""

    def __init__(self, search_products: list[dict], details: dict[str, dict] | None = None,
                 *, fail_batches: bool = False) -> None:
        self.search_products = search_products
        self.details = details or {}
        self.fail_batches = fail_batches
        self.search_calls: list[dict] = []
        self.detail_calls: list[list[str]] = []
        self.calls = 0

    def search(self, query, sort=None, limit=None):
        self.search_calls.append({"query": query, "sort": sort, "limit": limit})
        self.calls += 1
        return A2AResult(ARTIFACT_SEARCH, {"query": query,
                                          "products": list(self.search_products)})

    def detail(self, item_ids):
        self.detail_calls.append(list(item_ids))
        self.calls += 1
        if self.fail_batches:
            raise TaobaoA2AError("模拟批次失败")
        items = [self.details[i] for i in item_ids if i in self.details]
        return A2AResult(ARTIFACT_DETAIL, {"items": items, "itemCount": len(items)})


def make_search_item(item_id: str, **overrides) -> dict:
    item = {"itemId": item_id, "title": f"商品{item_id}", "price": "10",
            "shopName": "测试店", "procity": "浙江 杭州", "isAd": False}
    item.update(overrides)
    return item


def make_detail(item_id: str, **overrides) -> dict:
    detail = {"itemId": item_id, "title": f"商品{item_id}", "price": "25",
              "url": f"https://item.taobao.com/item.htm?id={item_id}",
              "brandName": "测试品牌", "shopName": "测试店"}
    detail.update(overrides)
    return detail


# --------------------------------------------------------------------------- #
# 文本处理
# --------------------------------------------------------------------------- #

def test_strip_highlight_removes_span_tags():
    assert strip_highlight(
        "无线蓝牙<span class=H>耳机</span><span class=H>头戴式</span>降噪"
    ) == "无线蓝牙耳机头戴式降噪"
    assert strip_highlight(None) == ""
    assert strip_highlight("无标签") == "无标签"


def test_strip_highlight_on_real_fixture_title():
    title = strip_highlight(fixture_search_products()[0]["title"])
    assert "<span" not in title and "保温" in title


@pytest.mark.parametrize(
    "text,expected",
    [
        ("【直饮款】初樱粉 240mL 【150g小巧便携+316L内胆】", 0.15),  # 选 150g，不选 240mL
        ("250毫升小样 5g", 0.005),          # 选 5g，不选 250毫升
        ("咖啡豆 250g 新鲜烘焙", 0.25),
        ("净重 1.2kg 装", 1.2),
        ("折叠椅 3.2kg", 3.2),
        ("2斤装 精选", 1.0),
        ("500mg 精华", 0.0005),
        # 只有容量 → 必须拒绝，不能拿容量当重量
        ("保温杯 500ml 大容量", None),
        ("容量 1.5L 大号", None),
        ("240mL", None),
        ("316L不锈钢 12小时保温", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_weight_kg(text, expected):
    result = extract_weight_kg(text)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_extract_weight_rejects_absurd_values():
    """>50kg 的匹配视为误匹配，宁可返回 None。"""
    assert extract_weight_kg("整箱 500kg 起批") is None


def test_extract_weight_scans_multiple_texts_in_order():
    assert extract_weight_kg(None, "", "第二个才有 300g") == pytest.approx(0.3)


def test_rank_to_heat_bounds_and_monotonic():
    assert rank_to_heat(1, 50) == RANK_HEAT_TOP
    assert rank_to_heat(50, 50) == RANK_HEAT_BOTTOM
    values = [rank_to_heat(r, 50) for r in range(1, 51)]
    assert values == sorted(values, reverse=True)
    assert all(RANK_HEAT_BOTTOM <= v <= RANK_HEAT_TOP for v in values)


def test_rank_to_heat_handles_single_result():
    assert rank_to_heat(1, 1) == RANK_HEAT_TOP
    assert rank_to_heat(1, 0) == RANK_HEAT_TOP


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #

def test_client_sends_tasks_send_envelope():
    session = FakeSession(FakeResponse(completed(ARTIFACT_SEARCH, {"products": []})))
    client = TaobaoA2AClient("https://example.test/a2a/x", interval=0, session=session)
    client.search("保温杯", sort="sales_desc", limit=5)

    body = json.loads(session.calls[0]["data"].decode("utf-8"))
    assert body["method"] == "tasks/send"
    assert body["jsonrpc"] == "2.0"
    part = body["params"]["message"]["parts"][0]
    assert part["type"] == "data"
    assert part["data"] == {"skillId": SKILL_SEARCH, "query": "保温杯",
                            "sort": "sales_desc", "limit": 5}


def test_client_unwraps_nested_data():
    session = FakeSession(FakeResponse(completed(ARTIFACT_SEARCH,
                                                 {"products": [{"itemId": "1"}]})))
    client = TaobaoA2AClient(interval=0, session=session)
    result = client.search("x")
    assert result.artifact == ARTIFACT_SEARCH
    assert result.data["products"][0]["itemId"] == "1"


def test_client_handles_artifact_without_data_part():
    """item-compare 的 artifact 可能只有渲染卡片、没有 data，不能炸。"""
    payload = {"result": {"task": {
        "status": {"state": "TASK_STATE_COMPLETED"},
        "artifacts": [{"name": "a2a-compare-card",
                       "parts": [{"text": "对比卡片"}]}],
    }}}
    client = TaobaoA2AClient(interval=0, session=FakeSession(FakeResponse(payload)))
    result = client.call({"skillId": "item-compare"})
    assert result.data == {}
    assert result.products() == []


def test_client_raises_on_task_failed_with_server_message():
    session = FakeSession(FakeResponse(failed("itemIds 数量需在 1-10 个之间，当前 0 个")))
    client = TaobaoA2AClient(interval=0, retries=0, session=session)
    with pytest.raises(TaobaoTaskFailed) as excinfo:
        client.detail(["x"])
    assert excinfo.value.state == "TASK_STATE_FAILED"
    assert "1-10" in excinfo.value.message


def test_client_does_not_retry_business_failure():
    session = FakeSession(FakeResponse(failed("unsupported request")))
    client = TaobaoA2AClient(interval=0, retries=3, session=session)
    with pytest.raises(TaobaoTaskFailed):
        client.call({"a": 1})
    assert len(session.calls) == 1, "业务拒绝不该重试"


def test_client_retries_transient_network_errors():
    session = FakeSession(
        ConnectionError("boom"),
        ConnectionError("boom"),
        FakeResponse(completed(ARTIFACT_SEARCH, {"products": []})),
    )
    client = TaobaoA2AClient(interval=0, retries=2, session=session)
    client.search("x")
    assert len(session.calls) == 3


def test_client_gives_up_after_retries():
    session = FakeSession(ConnectionError("boom"), ConnectionError("boom"),
                          ConnectionError("boom"))
    client = TaobaoA2AClient(interval=0, retries=2, session=session)
    with pytest.raises(TaobaoA2AError, match="已重试"):
        client.search("x")


def test_client_surfaces_jsonrpc_error():
    session = FakeSession(FakeResponse({"jsonrpc": "2.0", "error": {
        "code": -32601, "message": "Method not found: message/send"}}))
    client = TaobaoA2AClient(interval=0, retries=0, session=session)
    with pytest.raises(TaobaoA2AError, match="JSON-RPC"):
        client.search("x")


def test_client_surfaces_bad_json():
    session = FakeSession(FakeResponse(None, text="<html>", raise_on_json=True))
    client = TaobaoA2AClient(interval=0, retries=0, session=session)
    with pytest.raises(TaobaoA2AError, match="不是合法 JSON"):
        client.search("x")


def test_client_validates_detail_batch_size():
    client = TaobaoA2AClient(interval=0, session=FakeSession())
    with pytest.raises(ValueError, match=f"最多 {MAX_DETAIL_IDS}"):
        client.detail([str(i) for i in range(MAX_DETAIL_IDS + 1)])
    with pytest.raises(ValueError, match="不能为空"):
        client.detail([])


def test_client_validates_compare_count():
    client = TaobaoA2AClient(interval=0, session=FakeSession())
    with pytest.raises(ValueError, match="2-5"):
        client.compare(["1"])
    with pytest.raises(ValueError, match="2-5"):
        client.compare([str(i) for i in range(MAX_COMPARE_IDS + 1)])


def test_agent_card_reads_public_url():
    card = {"name": "ItemSearchAgent", "skills": [{"id": "item-search"}]}
    session = FakeSession(FakeResponse(card))
    client = TaobaoA2AClient("https://example.test/a2a/x", interval=0, session=session)
    assert client.agent_card()["name"] == "ItemSearchAgent"
    assert session.calls[0]["url"].endswith("/.well-known/agent.json")


# --------------------------------------------------------------------------- #
# 映射：价格陷阱
# --------------------------------------------------------------------------- #

def test_to_product_prefers_detail_price_over_search_price():
    """真实数据：搜索价 44.9，详情价 79.9。必须用后者。"""
    search_item = fixture_search_products()[0]
    detail = fixture_detail_item()
    assert float(search_item["price"]) == pytest.approx(44.9)
    assert float(detail["price"]) == pytest.approx(79.9)

    product = to_product(search_item, detail, query="保温杯", rank=1, window=50)
    assert product.price == pytest.approx(79.9)
    assert "真实售价" in product.note


def test_to_product_flags_search_price_when_detail_missing():
    """拿不到详情时必须显式标注，不能静默充当真实售价。"""
    product = to_product(fixture_search_products()[0], None, query="保温杯")
    assert product.price == pytest.approx(44.9)
    assert "⚠️" in product.note
    assert "来自搜索页" in product.note


def test_to_product_coerces_string_price():
    """接口返回的 price 是字符串。"""
    assert isinstance(fixture_search_products()[0]["price"], str)
    product = to_product(make_search_item("1"), make_detail("1", price="19.90"))
    assert product.price == pytest.approx(19.9)


def test_to_product_handles_unparseable_price():
    product = to_product(make_search_item("1", price="面议"), make_detail("1", price=None))
    assert product.price == 0.0


# --------------------------------------------------------------------------- #
# 映射：其他字段
# --------------------------------------------------------------------------- #

def test_to_product_maps_real_fixture_fields():
    product = to_product(fixture_search_products()[0], fixture_detail_item(),
                         query="保温杯", rank=1, window=50)
    assert product.external_id == "1060199595825"
    assert product.category == "保温杯"
    assert product.source == "taobao"
    assert product.url.startswith("https://item.taobao.com/")
    assert product.weight_kg == pytest.approx(0.15)
    assert "COOKER" in product.note
    assert "炊大皇官方旗舰店" in product.note
    assert "浙江 金华" in product.note


def test_to_product_uses_detail_url_over_tracking_url():
    search = make_search_item("1", auctionURL="http://click.simba.taobao.com/cc_im?x=1")
    detail = make_detail("1", url="https://item.taobao.com/item.htm?id=1")
    assert to_product(search, detail).url.startswith("https://item.taobao.com/")


def test_to_product_falls_back_to_tracking_url():
    search = make_search_item("1", auctionURL="http://click.simba.taobao.com/cc_im?x=1")
    assert "click.simba" in to_product(search, None).url


def test_to_product_strips_highlight_from_title():
    search = make_search_item("1", title="<span class=H>保温</span>杯")
    # 详情里的标题更权威；两边都要去高亮
    assert to_product(search, None).title == "保温杯"
    assert to_product(search, make_detail("1", title="<span class=H>保温</span>杯 正品")
                      ).title == "保温杯 正品"


def test_to_product_defaults_category_to_query():
    assert to_product(make_search_item("1"), None, query="降噪耳机").category == "降噪耳机"
    assert to_product(make_search_item("1"), None).category == "淘宝"


def test_to_product_category_override():
    product = to_product(make_search_item("1"), None, query="耳机", category="数码")
    assert product.category == "数码"


def test_to_product_unmapped_dimensions_are_neutral_defaults():
    product = to_product(make_search_item("1"), make_detail("1"))
    assert product.competition == 50.0
    assert product.repurchase == 50.0
    assert product.compliance_risk == 20.0
    assert product.virality == 50.0
    assert product.cost == 0.0, "淘宝是零售价，成本需另有来源"


def test_to_product_neutral_heat_mode():
    product = to_product(make_search_item("1"), make_detail("1"), heat_mode="neutral")
    assert product.heat == 50.0
    assert "中性值" in product.note


def test_to_product_weight_from_item_properties_first():
    detail = make_detail("1", defaultSkuName="忽略 999g",
                         itemProperties={"净重": "1.5kg", "品牌": "x"})
    assert to_product(make_search_item("1"), detail).weight_kg == pytest.approx(1.5)


def test_to_product_weight_falls_back_to_default_when_absent():
    product = to_product(make_search_item("1"), make_detail("1"))
    assert product.weight_kg == 0.5
    assert "取缺省值" in product.note


def test_to_product_can_disable_weight_extraction():
    detail = make_detail("1", defaultSkuName="500g")
    assert to_product(make_search_item("1"), detail, extract_weight=False).weight_kg == 0.5


def test_to_product_includes_properties_and_subtitle_in_note():
    detail = make_detail("1", subTitle="品质保证",
                         itemProperties={"杯子种类": "保温杯", "品牌": "忽略我"})
    note = to_product(make_search_item("1"), detail).note
    assert "杯子种类=保温杯" in note
    assert "品质保证" in note
    assert "品牌=忽略我" not in note


def test_to_product_note_within_length_limit():
    detail = make_detail("1", itemProperties={f"属性{i}": "值" * 40 for i in range(15)})
    assert len(to_product(make_search_item("1"), detail).note) <= 1000


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

def test_source_drops_ads_by_default():
    products = [make_search_item("1", isAd=True), make_search_item("2", isAd=False)]
    fake = FakeClient(products, {"2": make_detail("2")})
    source = TaobaoSource(["保温杯"], client=fake)
    result = source.fetch()

    assert [p.external_id for p in result] == ["2"]
    assert source.report.dropped_ads == 1


def test_source_can_keep_ads():
    products = [make_search_item("1", isAd=True)]
    fake = FakeClient(products, {"1": make_detail("1")})
    source = TaobaoSource(["x"], drop_ads=False, client=fake)
    assert len(source.fetch()) == 1


def test_source_drops_items_without_detail_by_default():
    products = [make_search_item("1"), make_search_item("2")]
    fake = FakeClient(products, {"1": make_detail("1")})  # 2 没有详情
    source = TaobaoSource(["x"], client=fake)
    result = source.fetch()

    assert [p.external_id for p in result] == ["1"]
    assert source.report.dropped_no_detail == 1
    assert any("拿不到 item-detail" in w for w in source.report.warnings)


def test_source_can_keep_items_without_detail_when_asked():
    products = [make_search_item("1")]
    fake = FakeClient(products, {})
    source = TaobaoSource(["x"], require_detail=False, client=fake)
    result = source.fetch()
    assert len(result) == 1
    assert "⚠️" in result[0].note, "保留时也必须标注价格来源不可靠"


def test_source_dedupes_across_queries():
    products = [make_search_item("1"), make_search_item("2")]
    fake = FakeClient(products, {"1": make_detail("1"), "2": make_detail("2")})
    source = TaobaoSource(["保温杯", "水杯"], client=fake)
    result = source.fetch()

    assert len(result) == 2
    assert source.report.duplicates == 2  # 第二个关键词的两条都重复
    assert len(fake.search_calls) == 2


def test_source_respects_detail_batch_size():
    products = [make_search_item(str(i)) for i in range(25)]
    details = {str(i): make_detail(str(i)) for i in range(25)}
    fake = FakeClient(products, details)
    source = TaobaoSource(["x"], detail_batch=10, client=fake)
    source.fetch()

    assert [len(c) for c in fake.detail_calls] == [10, 10, 5]
    assert all(len(c) <= MAX_DETAIL_IDS for c in fake.detail_calls)


def test_source_clamps_oversized_detail_batch():
    source = TaobaoSource(["x"], detail_batch=99, client=FakeClient([]))
    assert source.detail_batch == MAX_DETAIL_IDS


def test_source_max_detail_limits_calls():
    products = [make_search_item(str(i)) for i in range(20)]
    details = {str(i): make_detail(str(i)) for i in range(20)}
    fake = FakeClient(products, details)
    source = TaobaoSource(["x"], max_detail=5, client=fake)
    result = source.fetch()

    assert len(result) == 5
    assert sum(len(c) for c in fake.detail_calls) == 5


def test_source_isolates_failed_detail_batch():
    """单批详情失败不能中断整个拉取。"""
    products = [make_search_item(str(i)) for i in range(15)]
    fake = FakeClient(products, {}, fail_batches=True)
    source = TaobaoSource(["x"], detail_batch=10, client=fake)
    result = source.fetch()

    assert result == []
    batch_warnings = [w for w in source.report.warnings if "批次失败" in w]
    assert len(batch_warnings) == 2, "15 个商品按每批 10 个 → 两批都失败，各记一条"
    assert source.report.dropped_no_detail == 15


def test_source_passes_sort_and_limit():
    fake = FakeClient([])
    TaobaoSource(["x"], sort="price_asc", limit=7, client=fake).fetch()
    assert fake.search_calls[0] == {"query": "x", "sort": "price_asc", "limit": 7}


def test_source_normalises_invalid_sort():
    source = TaobaoSource(["x"], sort="乱写", client=FakeClient([]))
    assert source.sort == "sales_desc"


def test_source_requires_queries():
    with pytest.raises(ValueError, match="关键词"):
        TaobaoSource([], client=FakeClient([])).fetch()


def test_source_empty_recall_warns():
    source = TaobaoSource(["冷门词"], client=FakeClient([]))
    assert source.fetch() == []
    assert any("没有召回" in w for w in source.report.warnings)


def test_source_heat_uses_rank_within_window():
    products = [make_search_item(str(i)) for i in range(3)]
    details = {str(i): make_detail(str(i)) for i in range(3)}
    source = TaobaoSource(["x"], client=FakeClient(products, details))
    result = source.fetch()

    heats = [p.heat for p in result]
    assert heats == sorted(heats, reverse=True)
    assert heats[0] == RANK_HEAT_TOP


def test_source_report_summary_is_readable():
    products = [make_search_item("1", isAd=True), make_search_item("2")]
    source = TaobaoSource(["x"], client=FakeClient(products, {"2": make_detail("2")}))
    source.fetch()
    summary = source.report.summary()
    assert "召回 2 个" in summary and "过滤广告 1 个" in summary and "入库 1 个" in summary


def test_source_warns_that_margin_dimension_saturates():
    """淘宝没有成本 → 毛利率恒为 100%。这是会影响结论的事，必须主动提醒。"""
    products = [make_search_item("1")]
    source = TaobaoSource(["x"], client=FakeClient(products, {"1": make_detail("1")}))
    result = source.fetch()

    assert result[0].cost == 0.0
    assert source.report.notes, "沉默地让权重最高的维度失效是不可接受的"
    assert "毛利率" in source.report.notes[0]
    assert "1688" in source.report.notes[0]


def test_source_no_notes_when_nothing_recalled():
    source = TaobaoSource(["x"], client=FakeClient([]))
    source.fetch()
    assert source.report.notes == []


# --------------------------------------------------------------------------- #
# 真实响应 fixture 的形状守卫
# --------------------------------------------------------------------------- #

def test_search_fixture_shape_is_stable():
    """真实响应结构变了要立刻知道，否则解析会静默返回空。"""
    data = load_fixture("taobao_search.json")
    artifact = data["result"]["task"]["artifacts"][0]
    assert data["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert artifact["name"] == ARTIFACT_SEARCH
    inner = artifact["parts"][0]["data"]["data"]
    assert {"skillId", "products"} <= set(inner)
    product = inner["products"][0]
    assert {"itemId", "title", "price", "shopName", "procity", "isAd"} <= set(product)


def test_detail_fixture_shape_is_stable():
    data = load_fixture("taobao_detail.json")
    artifact = data["result"]["task"]["artifacts"][0]
    assert artifact["name"] == ARTIFACT_DETAIL
    inner = artifact["parts"][0]["data"]["data"]
    assert {"itemIds", "items", "itemCount"} <= set(inner)
    item = inner["items"][0]
    assert {"itemId", "title", "price", "itemProperties", "skuNames"} <= set(item)
    assert isinstance(item["itemProperties"], dict)
