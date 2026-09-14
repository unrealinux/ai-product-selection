"""抖音（抖店）数据源测试。

签名部分直接使用**官方文档给出的样例字符串**做断言，确保实现与文档一致：
https://op.jinritemai.com/docs/guide-docs/148/814 第六节
"""

from __future__ import annotations

import json

import pytest

from app import db
from app.crawler import get_source
from app.models import ProductIn
from app.scoring import profit_margin, rule_score
from app.sources.douyin import (
    ERROR_CODES,
    MAX_PAGE_SIZE,
    DouyinAPIError,
    DouyinClient,
    DouyinConfigError,
    DouyinSource,
    build_sign_string,
    canonical_param_json,
    commission_to_virality,
    in_sale_count_to_competition,
    method_to_path,
    sales_to_heat,
    sign_params,
    to_product,
)

# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #

#: 官方文档第六节样例值
DOC_SECRET = "749698a6-fcb3-4358-b241-ec1d93cf9c1f"
DOC_APP_KEY = "6844048284663924231"
DOC_PARAM_JSON = '{"page":"0","size":"20"}'
DOC_TIMESTAMP = "2020-07-05 22:33:59"
DOC_SIGN_STRING = (
    DOC_SECRET
    + "app_key6844048284663924231"
    + "methodproduct.list"
    + 'param_json{"page":"0","size":"20"}'
    + "timestamp2020-07-05 22:33:59"
    + "v2"
    + DOC_SECRET
)

#: 官方文档接口返回样例（buyin.kolMaterialsProductsSearch）
RAW_PRODUCT = {
    "activity_id": "1",
    "commission_type": "1",
    "cos_fee": "1",
    "cos_ratio": "10",
    "coupon_price": "92",
    "cover": "https://example.com/cover.webp",
    "detail_url": "https://haohuo.jinritemai.com/views/product/item2?id=3400312511185213726",
    "first_cid": "1206",
    "in_stock": "1",
    "kol_ad_cos_fee": "10",
    "kol_ad_cos_ratio": "10.00",
    "kol_cos_fee": "10",
    "kol_cos_ratio": "10.00",
    "limit_min_sale": "false",
    "post_free": "true",
    "presell_type": "0",
    "price": "100",
    "product_id": "3400312511185213726",
    "sales": "1234",
    "second_cid": "1402",
    "sharable": "true",
    "shop_id": "2232",
    "shop_name": "maggie测试小铺",
    "third_cid": "2634",
    "title": "测试商品",
}


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


class FakeSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, params=None, data=None, headers=None, timeout=None):
        self.calls.append(
            {"method": "POST", "url": url, "params": params, "data": data,
             "headers": headers, "timeout": timeout}
        )
        return self.responses.pop(0)

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {"method": "GET", "url": url, "params": params, "data": None,
             "headers": headers, "timeout": timeout}
        )
        return self.responses.pop(0)


class FakeDouyinClient:
    """替身客户端，按顺序吐出预设分页。"""

    def __init__(self, *pages, access_token: str = "tok") -> None:
        self.pages = list(pages)
        self.calls: list[dict] = []
        self.access_token = access_token

    def search_products(self, **kwargs):
        self.calls.append(kwargs)
        if not self.pages:
            return {"total": 0, "products": []}
        return self.pages.pop(0)


def make_client(*responses, **kwargs) -> tuple[DouyinClient, FakeSession]:
    session = FakeSession(*responses)
    kwargs.setdefault("access_token", "test-token")
    client = DouyinClient("appkey1234567890123", "secret-value",
                          session=session, **kwargs)
    return client, session


# --------------------------------------------------------------------------- #
# 签名：与官方文档样例逐字对比
# --------------------------------------------------------------------------- #

def test_sign_string_matches_official_doc_example():
    """官方文档第六节 step1~step4 的拼接结果必须逐字一致。"""
    public_params = {
        "app_key": DOC_APP_KEY,
        "method": "product.list",
        "param_json": DOC_PARAM_JSON,
        "timestamp": DOC_TIMESTAMP,
        "v": "2",
    }
    assert build_sign_string(public_params, DOC_SECRET) == DOC_SIGN_STRING


def test_sign_excludes_access_token_and_sign_method():
    """官方：access_token 和 sign_method 不参与加密。"""
    base = {
        "app_key": DOC_APP_KEY,
        "method": "product.list",
        "param_json": DOC_PARAM_JSON,
        "timestamp": DOC_TIMESTAMP,
        "v": "2",
    }
    with_extras = dict(base, access_token="abc", sign_method="hmac-sha256", sign="deadbeef")
    assert build_sign_string(with_extras, DOC_SECRET) == build_sign_string(base, DOC_SECRET)
    assert sign_params(with_extras, DOC_SECRET) == sign_params(base, DOC_SECRET)


def test_sign_methods_are_lowercase_hex_of_expected_length():
    params = {"app_key": DOC_APP_KEY, "method": "product.list",
              "param_json": DOC_PARAM_JSON, "timestamp": DOC_TIMESTAMP, "v": "2"}
    md5 = sign_params(params, DOC_SECRET, "md5")
    sha = sign_params(params, DOC_SECRET, "hmac-sha256")
    assert len(md5) == 32 and md5 == md5.lower()
    assert len(sha) == 64 and sha == sha.lower()
    assert md5 != sha


def test_sign_rejects_unknown_method():
    with pytest.raises(DouyinConfigError, match="不支持的签名算法"):
        sign_params({"a": "b"}, "s", "sha1")


def test_hmac_key_and_message_are_both_the_secret_wrapped_string():
    """官方 go 示例：key = app_secret，message = app_secret + ... + app_secret。"""
    import hashlib
    import hmac

    params = {"app_key": DOC_APP_KEY, "method": "product.list",
              "param_json": DOC_PARAM_JSON, "timestamp": DOC_TIMESTAMP, "v": "2"}
    expected = hmac.new(
        DOC_SECRET.encode("utf-8"), DOC_SIGN_STRING.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert sign_params(params, DOC_SECRET, "hmac-sha256") == expected


# --------------------------------------------------------------------------- #
# param_json 规范化
# --------------------------------------------------------------------------- #

def test_canonical_param_json_sorts_keys_and_removes_spaces():
    assert canonical_param_json({"size": "20", "page": "0"}) == '{"page":"0","size":"20"}'
    assert " " not in canonical_param_json({"a": "1", "b": "2"})


def test_canonical_param_json_escapes_special_chars():
    """官方要求 & < > 与 \\b 转义。"""
    out = canonical_param_json({"k": "a&b<c>d"})
    assert "\\u0026" in out and "\\u003c" in out and "\\u003e" in out
    assert "&" not in out and "<" not in out and ">" not in out


def test_canonical_param_json_stringifies_values_and_drops_none():
    out = canonical_param_json({"page": 1, "flag": True, "skip": None})
    assert out == '{"flag":"true","page":"1"}'


def test_canonical_param_json_nested_list():
    assert canonical_param_json({"first_cids": [1, 2]}) == '{"first_cids":["1","2"]}'


def test_method_to_path():
    assert method_to_path("buyin.kolMaterialsProductsSearch") == "/buyin/kolMaterialsProductsSearch"
    assert method_to_path("token.create") == "/token/create"


# --------------------------------------------------------------------------- #
# HTTP 客户端
# --------------------------------------------------------------------------- #

def test_call_puts_param_json_in_body_and_signs_query():
    payload = {"code": 10000, "msg": "success", "data": {"total": "1", "products": [RAW_PRODUCT]}}
    client, session = make_client(FakeResponse(payload))
    data = client.call("buyin.kolMaterialsProductsSearch",
                       {"page": 1, "page_size": 20, "search_type": 1, "sort_type": 1},
                       timestamp=DOC_TIMESTAMP)

    assert data["total"] == "1"  # call() 返回原始 data，类型归一化在 search_products 完成
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://openapi-fxg.jinritemai.com/buyin/kolMaterialsProductsSearch"
    # param_json 走 body，不在 query 里
    assert "param_json" not in call["params"]
    assert json.loads(call["data"]) == {
        "page": "1", "page_size": "20", "search_type": "1", "sort_type": "1"
    }
    # 其余公共参数齐备
    for key in ("method", "app_key", "timestamp", "v", "sign", "sign_method", "access_token"):
        assert key in call["params"], key
    assert call["params"]["timestamp"] == DOC_TIMESTAMP


def test_call_signature_is_verifiable_by_recompute():
    client, session = make_client(FakeResponse({"code": 10000, "data": {}}))
    client.call("buyin.kolMaterialsProductsSearch", {"page": 1}, timestamp=DOC_TIMESTAMP)
    query = dict(session.calls[0]["params"])

    body = session.calls[0]["data"].decode("utf-8")
    recomputed = sign_params(
        {**query, "param_json": body}, client.app_secret, client.sign_method
    )
    assert query["sign"] == recomputed


def test_call_raises_with_official_error_message():
    payload = {"code": 11, "msg": "", "data": {}}
    client, _ = make_client(FakeResponse(payload))
    with pytest.raises(DouyinAPIError) as excinfo:
        client.call("buyin.kolMaterialsProductsSearch", {"page": 1})
    assert excinfo.value.code == 11
    assert "签名校验失败" in str(excinfo.value)


def test_expired_token_error_flags_reauthorization():
    client, _ = make_client(FakeResponse({"code": 30002, "msg": "", "data": {}}))
    with pytest.raises(DouyinAPIError) as excinfo:
        client.call("buyin.kolMaterialsProductsSearch", {"page": 1})
    assert excinfo.value.expired_token is True


def test_call_requires_token_when_asked():
    session = FakeSession()
    client = DouyinClient("appkey1234567890123", "secret-value", access_token="", session=session)
    with pytest.raises(DouyinConfigError, match="access_token"):
        client.call("buyin.kolMaterialsProductsSearch", {"page": 1})


def test_call_wraps_non_json_response():
    client, _ = make_client(FakeResponse(None, status_code=502, text="<html>bad gateway</html>",
                                         raise_on_json=True))
    with pytest.raises(DouyinAPIError, match="不是合法 JSON"):
        client.call("buyin.kolMaterialsProductsSearch", {"page": 1})


def test_client_requires_app_key_and_secret():
    with pytest.raises(DouyinConfigError, match="app_key"):
        DouyinClient("", "", access_token="x")


def test_create_self_token_matches_official_doc_params():
    """文档 97/1896：param_json 为 {"code":"","grant_type":"authorization_self","shop_id":"..."}"""
    payload = {"code": 10000, "msg": "success",
               "data": {"access_token": "new-token-abc", "expires_in": "86400"}}
    client, session = make_client(FakeResponse(payload))
    data = client.create_self_token("12345678")

    call = session.calls[0]
    assert call["url"].endswith("/token/create")
    assert "access_token" not in call["params"], "token.create 不应携带 access_token"
    assert json.loads(call["data"]) == {
        "code": "", "grant_type": "authorization_self", "shop_id": "12345678"
    }
    assert json.loads(call["data"]).keys() == {"code", "grant_type", "shop_id"}
    assert data["access_token"] == "new-token-abc"
    assert client.access_token == "new-token-abc"


def test_create_self_token_requires_shop_id():
    client, _ = make_client(FakeResponse({"code": 10000, "data": {}}))
    with pytest.raises(DouyinConfigError, match="shop_id"):
        client.create_self_token()


def test_search_products_rejects_oversized_page_size():
    client, _ = make_client(FakeResponse({"code": 10000, "data": {}}))
    with pytest.raises(ValueError, match="page_size"):
        client.search_products(page_size=MAX_PAGE_SIZE + 1)


# --------------------------------------------------------------------------- #
# 维度映射
# --------------------------------------------------------------------------- #

def test_sales_to_heat_is_monotonic_and_bounded():
    assert sales_to_heat(0) == 0.0
    assert sales_to_heat(-5) == 0.0
    values = [sales_to_heat(n) for n in (1, 100, 1_000, 10_000, 100_000, 1_000_000)]
    assert values == sorted(values)
    assert all(0 <= v <= 100 for v in values)
    assert sales_to_heat(1_000_000) == pytest.approx(100.0)


def test_competition_grows_with_market_size():
    assert in_sale_count_to_competition(0) == 0.0
    assert in_sale_count_to_competition(100) < in_sale_count_to_competition(100_000)
    assert all(
        0 <= in_sale_count_to_competition(n) <= 100
        for n in (0, 10, 1_000, 1_000_000, 10_000_000)
    )


def test_commission_to_virality_floor_and_ceiling():
    assert commission_to_virality(0.0) == 30.0
    assert commission_to_virality(0.30) == 100.0
    assert commission_to_virality(0.50) == 100.0


def test_to_product_unit_conversion_fen_to_yuan():
    """官方：price / kol_cos_fee 单位为分，kol_cos_ratio 需除以 100。"""
    product = to_product(RAW_PRODUCT, competition_total=1000)

    assert isinstance(product, ProductIn)
    assert product.price == pytest.approx(1.00)     # 100 分
    assert product.cost == pytest.approx(0.90)      # 100 - 10 分
    assert product.title == "测试商品"
    assert product.source == "douyin"
    assert product.url == RAW_PRODUCT["detail_url"]


def test_to_product_margin_equals_commission_ratio():
    """分销视角：规则引擎算出的毛利率应等于达人佣金率（10.00 → 10%）。"""
    product = to_product(RAW_PRODUCT, competition_total=1000)
    assert profit_margin(product.price, product.cost) == pytest.approx(0.10, abs=1e-6)

    scored = rule_score(product)
    assert scored.profit_margin == pytest.approx(0.10, abs=1e-6)
    assert 0 <= scored.total <= 100


def test_to_product_uses_ratio_when_fee_missing():
    raw = dict(RAW_PRODUCT)
    raw.pop("kol_cos_fee")
    product = to_product(raw, competition_total=1000)
    assert profit_margin(product.price, product.cost) == pytest.approx(0.10, abs=1e-6)


def test_to_product_defaults_are_documented_in_note():
    product = to_product(RAW_PRODUCT, competition_total=1000)
    assert product.weight_kg == 0.5
    assert product.repurchase == 50.0
    assert product.compliance_risk == 20.0
    assert "人工复核" in product.note
    assert "历史销量 1234" in product.note


def test_to_product_handles_empty_and_dirty_values():
    product = to_product({"title": "  ", "price": "", "sales": None, "product_id": "9"})
    assert product.title == "抖音商品9"
    assert product.price == 0.0
    assert product.heat == 0.0
    assert product.category == "未分类"


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

def test_source_paginates_and_dedupes():
    page1 = {"total": 1000, "products": [
        RAW_PRODUCT, dict(RAW_PRODUCT, product_id="2", title="商品二")
    ]}
    page2 = {"total": 1000, "products": [
        dict(RAW_PRODUCT, title="重复商品")  # 同一个 product_id
    ]}
    fake = FakeDouyinClient(page1, page2)
    source = DouyinSource(["咖啡"], page_size=2, max_pages=5, client=fake)

    products = source.fetch()
    # product_id 3400312511185213726 出现两次，只保留第一条；加上商品二，共 2 条
    assert len(products) == 2
    assert {p.title for p in products} == {"测试商品", "商品二"}
    # page_size=2 且第 2 页只有 1 条 < page_size → 停止，不会请求第 3 页
    assert len(fake.calls) == 2


def test_source_filters_out_of_stock_and_unsharable():
    page = {"total": 10, "products": [
        dict(RAW_PRODUCT, product_id="1", in_stock="0"),
        dict(RAW_PRODUCT, product_id="2", sharable="false"),
        dict(RAW_PRODUCT, product_id="3"),
    ]}
    source = DouyinSource(["x"], page_size=20, client=FakeDouyinClient(page))
    products = source.fetch()
    assert [p.title for p in products] == ["测试商品"]
    assert "product_id" not in products[0].note or "3" in products[0].note


def test_source_empty_keyword_defaults_to_full_market():
    source = DouyinSource([], client=FakeDouyinClient())
    assert source.keywords == [""]


def test_source_passes_filters_to_client():
    fake = FakeDouyinClient({"total": 0, "products": []})
    source = DouyinSource(["咖啡"], page_size=5, max_pages=1, search_type=2,
                          sort_type=0, first_cids=[1206], cos_ratio_min=100, client=fake)
    source.fetch()
    call = fake.calls[0]
    assert call["title"] == "咖啡"
    assert call["search_type"] == 2 and call["sort_type"] == 0
    assert call["first_cids"] == [1206]
    assert call["cos_ratio_min"] == 100


# --------------------------------------------------------------------------- #
# 与 crawler 注册表的集成
# --------------------------------------------------------------------------- #

def test_get_source_builds_douyin_source_from_keyword_string():
    source = get_source("douyin", "咖啡,保温杯", {"page_size": 5, "max_pages": 1})
    assert isinstance(source, DouyinSource)
    assert source.keywords == ["咖啡", "保温杯"]
    assert source.page_size == 5


def test_get_source_unknown_name_lists_douyin():
    with pytest.raises(KeyError, match="douyin"):
        get_source("nope")


def test_error_codes_table_covers_documented_entries():
    for code in (6, 11, 30002, 30007):
        assert code in ERROR_CODES


# --------------------------------------------------------------------------- #
# 落库集成：拉取 → 入库 → 规则打分
# --------------------------------------------------------------------------- #

def test_end_to_end_fetch_store_and_score(tmp_path):
    """全链路：抖音响应 → ProductIn → SQLite → 打分引擎。"""
    db_path = tmp_path / "douyin.db"
    db.init_db(db_path)

    page = {"total": 5000, "products": [
        RAW_PRODUCT,
        dict(RAW_PRODUCT, product_id="999", title="保温杯", price="2000",
             kol_cos_fee="600", kol_cos_ratio="30.00", sales="50000"),
    ]}
    source = DouyinSource(["杯子"], client=FakeDouyinClient(page))

    saved = db.bulk_upsert(source.fetch(), db_path)
    assert len(saved) == 2

    # 再次导入同一批数据应去重更新而非新增
    again = DouyinSource(["杯子"], client=FakeDouyinClient(page)).fetch()
    assert len(db.bulk_upsert(again, db_path)) == 2

    by_title = {item.title: item for item in db.list_products(db_path=db_path)}
    assert len(by_title) == 2

    cheap = by_title["测试商品"]
    expensive = by_title["保温杯"]

    # 售价 20.00 元、佣金 6.00 元 → 毛利率 30%
    assert expensive.price == pytest.approx(20.0)
    assert expensive.cost == pytest.approx(14.0)
    assert profit_margin(expensive.price, expensive.cost) == pytest.approx(0.30)

    # 销量更高、佣金更高的商品应得分更高
    assert rule_score(expensive).total > rule_score(cheap).total
    assert rule_score(expensive).grade in {"S", "A", "B"}

    # 榜单能读到
    assert len(db.latest_scores(db_path=db_path)) == 0  # 尚未打分入库
    raw_result = rule_score(expensive)
    db.save_score(
        {
            "product_id": expensive.id,
            "total": raw_result.total,
            "grade": raw_result.grade,
            "dimensions": raw_result.dimensions,
            "profit_margin": raw_result.profit_margin,
            "advice": raw_result.advice,
        },
        db_path,
    )
    scores = db.latest_scores(db_path=db_path)
    assert len(scores) == 1
    assert scores[0]["title"] == "保温杯"
    assert scores[0]["dimensions"]["margin"] > 0
