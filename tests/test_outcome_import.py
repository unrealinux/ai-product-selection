"""真实结果表的列识别与商品匹配测试。

真实导出表（生意参谋 / 抖店 / 淘宝订单）有两个坑，这里逐个盯住：

1. **「商品ID」是平台 ID，不是本库主键** —— 只按主键匹配会整批失败
2. **标题带促销后缀** —— 只按精确匹配会大面积丢行，模糊匹配要有阈值与歧义提示
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.models import Product
from app.outcome_import import (
    DEFAULT_TITLE_THRESHOLD,
    ProductIndex,
    build_column_map,
)


def product(pid: int, title: str, *, external_id: str = "", price: float = 99.0,
            cost: float = 30.0) -> Product:
    return Product(
        id=pid, title=title, external_id=external_id, category="家居",
        price=price, cost=cost, source="taobao", url="", image_url="",
        heat=50.0, competition=50.0, weight_kg=0.5, repurchase=50.0,
        compliance_risk=20.0, virality=50.0, note="",
        created_at=datetime(2024, 1, 1, 0, 0, 0),
    )


# --------------------------------------------------------------------------- #
# 列识别
# --------------------------------------------------------------------------- #

def test_maps_real_export_headers():
    columns = ["商品ID", "商品名称", "统计日期", "商品曝光", "商品点击",
               "支付件数", "支付金额", "推广花费", "成功退款笔数", "采购成本"]
    mapping = build_column_map(columns)

    assert mapping["商品ID"] == "product_id"
    assert mapping["商品名称"] == "title"
    assert mapping["统计日期"] == "window_date"
    assert mapping["商品曝光"] == "impressions"
    assert mapping["商品点击"] == "clicks"
    assert mapping["支付件数"] == "units"
    assert mapping["支付金额"] == "revenue"
    assert mapping["推广花费"] == "ad_spend"
    assert mapping["成功退款笔数"] == "returns"
    assert mapping["采购成本"] == "cogs"


def test_maps_platform_id_column_to_external_id():
    mapping = build_column_map(["平台商品ID", "宝贝标题", "销售额"])
    assert mapping["平台商品ID"] == "external_id"
    assert mapping["宝贝标题"] == "title"


def test_window_date_is_separate_from_start_end():
    mapping = build_column_map(["日期", "开始日期", "结束日期"])
    assert mapping["日期"] == "window_date"
    assert mapping["开始日期"] == "window_start"
    assert mapping["结束日期"] == "window_end"


# --------------------------------------------------------------------------- #
# 匹配优先级
# --------------------------------------------------------------------------- #

@pytest.fixture
def index() -> ProductIndex:
    return ProductIndex([
        product(1, "304不锈钢保温杯 500ml 便携", external_id="1060199595825"),
        product(2, "儿童磁性拼搭积木 100 片", external_id="A002"),
        product(3, "硅胶折叠收纳水杯 500ml"),
    ])


def test_matches_internal_primary_key_first(index):
    match = index.match(raw_id=1)
    assert match.product.id == 1
    assert match.key == "product_id"


def test_matches_platform_id_stored_as_external_id(index):
    """真实导出表的「商品ID」是平台 ID —— 必须能匹配上。"""
    match = index.match(raw_id="1060199595825")
    assert match.product.id == 1
    assert match.key == "external_id"


def test_non_numeric_id_is_tried_as_external_id(index):
    match = index.match(raw_id="A002")
    assert match.product.id == 2
    assert match.key == "external_id"


def test_explicit_external_id_column(index):
    match = index.match(external_id="1060199595825", title="完全对不上的标题")
    assert match.product.id == 1
    assert match.key == "external_id"


def test_exact_title_match(index):
    match = index.match(title="硅胶折叠收纳水杯 500ml")
    assert match.product.id == 3
    assert match.key == "title"


def test_fuzzy_title_match_above_threshold(index):
    """平台导出的标题常带促销后缀，精确匹配会失败。"""
    match = index.match(title="【爆款】304不锈钢保温杯 500ml 便携 包邮")
    assert match.product.id == 1
    assert match.key == "title_fuzzy"
    assert match.score >= DEFAULT_TITLE_THRESHOLD
    assert match.matched


def test_fuzzy_title_below_threshold_is_not_matched(index):
    match = index.match(title="完全不相干的东西 儿童玩具车")
    assert not match.matched
    assert match.key == "none"


def test_no_identifiers_means_no_match(index):
    assert not index.match().matched


def test_unknown_internal_id_falls_through_to_no_match(index):
    assert not index.match(raw_id=999).matched


# --------------------------------------------------------------------------- #
# 歧义
# --------------------------------------------------------------------------- #

def test_ambiguous_match_is_flagged():
    """两个商品标题非常接近时，必须标出歧义而不是静默选一个。"""
    index = ProductIndex([
        product(1, "304不锈钢保温杯 500ml"),
        product(2, "304不锈钢保温杯 500ml 便携"),
    ], title_threshold=0.45)
    match = index.match(title="304不锈钢保温杯 500ml")
    assert match.key == "title"  # 精确命中优先级更高

    fuzzy = index.match(title="不锈钢保温杯 304 500ml 大容量")
    assert fuzzy.matched
    assert fuzzy.key == "title_fuzzy"
    assert fuzzy.alternatives, "应把次优候选一并列出"
    assert fuzzy.ambiguous, "两个候选非常接近，应标记为歧义"


def test_candidate_limit_keeps_matching_bounded():
    index = ProductIndex(
        [product(i, f"商品标题第 {i} 号") for i in range(1, 500)],
        candidate_limit=10,
    )
    match = index.match(title="商品标题第 42 号")
    assert match.matched


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #

def test_describe_reports_the_key_used(index):
    assert index.match(raw_id=1).describe() == "库内主键"
    assert index.match(raw_id="1060199595825").describe() == "平台商品ID"
    assert index.match(title="硅胶折叠收纳水杯 500ml").describe() == "标题精确"
    assert "标题模糊" in index.match(title="304不锈钢保温杯 500ml 便携 新款").describe()
    assert index.match(title="毫不相干").describe() == "匹配不上"
