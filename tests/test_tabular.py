"""表格导入测试。

覆盖：编码/分隔符识别、列名自动映射、单位换算、量纲判定、
缺列推导、脏数据跳过、映射方案复用。

这些都是"算错了不会报错、只会静默给出错误商品"的地方。
"""

from __future__ import annotations

import csv
import io

import pytest

from app import db, service
from app.models import ProductIn
from app.scoring import profit_margin
from app.tabular import (
    ColumnMapping,
    RateScale,
    TableData,
    TabularSource,
    auto_map,
    build_products,
    build_provenance,
    column_fingerprint,
    detect_price_unit,
    detect_weight_unit,
    normalize_column,
    parse_number,
    rate_scale,
    read_csv,
    read_excel,
    read_table,
)

# --------------------------------------------------------------------------- #
# 造数据
# --------------------------------------------------------------------------- #

ALI_HEADER = ["商品标题", "商品ID", "一级类目", "批发价(元)", "30天成交", "复购率",
              "重量(g)", "商品链接"]
ALI_ROWS = [
    ["304不锈钢保温杯 500ml", "A001", "日用百货", "18.50", "12000", "35%", "320",
     "https://detail.1688.com/x1"],
    ["硅胶折叠水杯 户外便携", "A002", "户外运动", "9.9", "860", "0.42", "180",
     "https://detail.1688.com/x2"],
    ["", "A003", "杂物", "5", "10", "10%", "100", ""],              # 无标题 → 跳过
    ["玻璃保鲜盒 三件套", "A004", "厨房用品", "26", "3400", "18%", "900",
     "https://detail.1688.com/x3"],
]


@pytest.fixture
def ali_csv(tmp_path):
    """1688 风格的 GBK 编码 CSV。"""
    path = tmp_path / "1688导出.csv"
    with io.open(path, "w", encoding="gb18030", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ALI_HEADER)
        writer.writerows(ALI_ROWS)
    return path


@pytest.fixture
def utf8_tsv(tmp_path):
    path = tmp_path / "products.tsv"
    path.write_text(
        "title\tcost\tsales\tweight_kg\tcategory\n"
        "咖啡豆 500g\t32\t2000\t0.5\t食品\n"
        "保温杯\t18\t900\t0.32\t家居\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def test_normalize_column_strips_noise():
    assert normalize_column("批发价(元)") == "批发价元"
    assert normalize_column("Weight_kg") == "weightkg"
    assert normalize_column("  商品 标题 ") == "商品标题"
    assert normalize_column(None) == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("18.50", 18.5), ("1,234.5", 1234.5), ("￥26", 26.0), ("35%", 35.0),
        ("-3", -3.0), (12, 12.0), (3.5, 3.5),
        ("暂无", None), ("", None), ("-", None), (None, None), ("abc", None),
        (float("nan"), None),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_parse_number_rejects_bool():
    assert parse_number(True) is None


# --------------------------------------------------------------------------- #
# 列名识别
# --------------------------------------------------------------------------- #

def test_auto_map_recognises_chinese_and_english():
    mapping = auto_map(ALI_HEADER)
    assert mapping["title"] == "商品标题"
    assert mapping["external_id"] == "商品ID"
    assert mapping["category"] == "一级类目"
    assert mapping["cost"] == "批发价(元)"
    assert mapping["sales"] == "30天成交"
    assert mapping["repurchase"] == "复购率"
    assert mapping["weight_kg"] == "重量(g)"
    assert mapping["url"] == "商品链接"


def test_auto_map_english_headers():
    mapping = auto_map(["title", "purchase_price", "sold_count", "weight_kg"])
    assert mapping["title"] == "title"
    assert mapping["cost"] == "purchase_price"
    assert mapping["sales"] == "sold_count"
    assert mapping["weight_kg"] == "weight_kg"


def test_auto_map_does_not_reuse_one_column_for_two_fields():
    mapping = auto_map(["商品标题", "标题"])
    assert len(set(mapping.values())) == len(mapping)


def test_auto_map_falls_back_to_substring_match():
    mapping = auto_map(["1688商品标题（中文）", "批发价RMB"])
    assert mapping["title"] == "1688商品标题（中文）"
    assert mapping["cost"] == "批发价RMB"


def test_auto_map_empty():
    assert auto_map([]) == {}


@pytest.mark.parametrize(
    "column,expected",
    [
        ("重量(g)", "g"), ("重量(kg)", "kg"), ("毛重（克）", "g"),
        ("weight_g", "g"), ("weight_kg", "kg"), ("净重(mg)", "mg"),
        ("重量(斤)", "jin"),
        ("重量", None), ("weight", None), ("gross_weight", None),
    ],
)
def test_detect_weight_unit(column, expected):
    """关键：`weight` 里的 g 不能被误判成克。"""
    assert detect_weight_unit(column) == expected


@pytest.mark.parametrize(
    "column,expected",
    [
        ("批发价(元)", "yuan"), ("price_yuan", "yuan"), ("Price_CNY", "yuan"),
        ("rmb_price", "yuan"), ("单价(分)", "fen"), ("price_fen", "fen"),
        ("价格", None), ("cost", None),
        # 不能把「分钟」「分析」误判成分
        ("发货分钟数", None), ("分析结果", None),
    ],
)
def test_detect_price_unit(column, expected):
    assert detect_price_unit(column) == expected


def test_column_fingerprint_ignores_order_and_noise():
    a = column_fingerprint(["商品标题", "批发价(元)"])
    b = column_fingerprint(["批发价（元）", "商品标题 "])
    assert a == b
    assert a != column_fingerprint(["商品标题", "售价"])


# --------------------------------------------------------------------------- #
# 量纲判定
# --------------------------------------------------------------------------- #

def test_rate_scale_ratio_column_multiplied():
    assert rate_scale([0.35, 0.42, 0.18]) == 100.0


def test_rate_scale_percent_column_left_alone():
    assert rate_scale(["35%", "42%", "18%"]) == 1.0


def test_rate_scale_zero_to_hundred_left_alone():
    assert rate_scale([35, 62, 18]) == 1.0


def test_rate_scale_handles_mixed_percent_and_ratio_per_value():
    """同一列里既有 35% 又有 0.42 —— 按值判定，不能整列一个倍数。"""
    scale = RateScale.from_values(["35%", "0.42", "18%"])
    assert scale.apply("35%") == pytest.approx(35.0)
    assert scale.apply("0.42") == pytest.approx(42.0)
    assert scale.apply("18%") == pytest.approx(18.0)
    assert scale.rescaled is True
    assert scale.ambiguous is False


def test_rate_scale_flags_ambiguous_column_instead_of_guessing():
    """没有百分号又同时出现 ≤1 与 >1 —— 量纲真歧义，不能静默猜。"""
    scale = RateScale.from_values([0.62, 0.35, 62])
    assert scale.ambiguous is True
    assert scale.rescaled is False


def test_rate_scale_empty_and_all_missing():
    assert RateScale.from_values([]).plain_factor == 1.0
    assert RateScale.from_values(["暂无", "-"]).plain_factor == 1.0


def test_rate_scale_apply_returns_none_for_garbage():
    assert RateScale.from_values([50]).apply("abc") is None
    assert RateScale.from_values([50]).apply(None) is None


# --------------------------------------------------------------------------- #
# 读表
# --------------------------------------------------------------------------- #

def test_read_csv_detects_gbk_encoding(ali_csv):
    data = read_csv(ali_csv)
    assert data.encoding == "gb18030"
    assert data.columns == ALI_HEADER
    assert len(data) == 4
    assert data.rows[0]["商品标题"] == "304不锈钢保温杯 500ml"


def test_read_csv_detects_tab_delimiter(utf8_tsv):
    data = read_csv(utf8_tsv)
    assert data.delimiter == "\t"
    assert data.columns == ["title", "cost", "sales", "weight_kg", "category"]
    assert len(data) == 2


def test_read_csv_skips_blank_lines(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("title,cost\n甲,10\n\n乙,20\n,,\n", encoding="utf-8")
    data = read_csv(path)
    assert len(data) == 2


def test_read_csv_dedupes_and_fills_header_names(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("title,title,,cost\n甲,乙,丙,10\n", encoding="utf-8")
    data = read_csv(path)
    assert data.columns == ["title", "title_2", "列3", "cost"]


def test_read_table_dispatches_by_suffix(ali_csv, utf8_tsv):
    assert read_table(ali_csv).columns == ALI_HEADER
    assert read_table(utf8_tsv).delimiter == "\t"


def test_read_table_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "nope.csv")


def test_read_excel_round_trip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "商品"
    sheet.append(["商品标题", "批发价(元)", "重量(g)"])
    sheet.append(["陶瓷马克杯", 12.5, 380])
    sheet.append(["玻璃水壶", 20, 700])
    path = tmp_path / "商品表.xlsx"
    workbook.save(path)

    data = read_excel(path)
    assert data.sheet == "商品"
    assert data.columns == ["商品标题", "批发价(元)", "重量(g)"]
    assert len(data) == 2
    assert data.rows[0]["商品标题"] == "陶瓷马克杯"


def test_read_excel_specific_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "第一页"
    workbook.active.append(["title"])
    second = workbook.create_sheet("第二页")
    second.append(["title", "cost"])
    second.append(["乙商品", 5])
    path = tmp_path / "multi.xlsx"
    workbook.save(path)

    data = read_excel(path, sheet="第二页")
    assert data.sheet == "第二页"
    assert len(data) == 1
    with pytest.raises(ValueError, match="不存在"):
        read_excel(path, sheet="没有这页")


def test_read_excel_rejects_legacy_xls(tmp_path):
    path = tmp_path / "old.xls"
    path.write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(ValueError, match="另存为"):
        read_excel(path)


# --------------------------------------------------------------------------- #
# 组装商品
# --------------------------------------------------------------------------- #

def test_build_products_from_1688_style_csv(ali_csv):
    data = read_csv(ali_csv)
    mapping = ColumnMapping.auto(data.columns, source="1688")
    result = build_products(data, mapping)

    assert len(result.products) == 3      # 空标题那行被跳过
    assert result.skipped == 1

    cup = result.products[0]
    assert cup.title == "304不锈钢保温杯 500ml"
    assert cup.external_id == "A001"
    assert cup.category == "日用百货"
    assert cup.source == "1688"
    # 只有采购价 → 按加价 2.5 倍推导售价
    assert cup.cost == pytest.approx(18.5)
    assert cup.price == pytest.approx(46.25)
    assert profit_margin(cup.price, cup.cost) == pytest.approx(0.6)
    # 重量 (g) → kg
    assert cup.weight_kg == pytest.approx(0.32)
    # 复购率 35% 原样
    assert cup.repurchase == pytest.approx(35.0)
    # 30天成交 → 热度对数映射
    assert 0 < cup.heat < 100
    assert cup.weight_kg < 1, "克必须换算成千克，否则物流维度会错得离谱"


def test_build_products_mixed_rate_column_within_row(ali_csv):
    """同一列里 35% 与 0.42 都要被正确解释为 35 和 42。"""
    data = read_csv(ali_csv)
    mapping = ColumnMapping.auto(data.columns)
    products = build_products(data, mapping).products
    assert products[0].repurchase == pytest.approx(35.0)
    assert products[1].repurchase == pytest.approx(42.0)
    assert products[2].repurchase == pytest.approx(18.0)


def test_build_products_warns_on_ambiguous_column():
    table = TableData(
        columns=["商品标题", "批发价(元)", "竞争指数"],
        rows=[
            {"商品标题": "甲", "批发价(元)": "10", "竞争指数": "0.62"},
            {"商品标题": "乙", "批发价(元)": "20", "竞争指数": "62"},
        ],
        path="inline",
    )
    mapping = ColumnMapping.auto(table.columns)
    result = build_products(table, mapping)
    assert any("量纲不一致" in warning for warning in result.warnings)


def test_build_products_uses_explicit_price_over_derivation():
    table = TableData(
        columns=["商品标题", "售价", "成本"],
        rows=[{"商品标题": "甲", "售价": "100", "成本": "40"}],
    )
    result = build_products(table, ColumnMapping.auto(table.columns))
    product = result.products[0]
    assert product.price == pytest.approx(100.0)
    assert product.cost == pytest.approx(40.0)


def test_build_products_clamps_cost_above_price():
    table = TableData(
        columns=["商品标题", "售价", "成本"],
        rows=[{"商品标题": "倒挂", "售价": "10", "成本": "99"}],
    )
    product = build_products(table, ColumnMapping.auto(table.columns)).products[0]
    assert product.cost == pytest.approx(10.0), "成本高于售价时封顶，避免负毛利"


def test_build_products_fen_price_unit():
    table = TableData(
        columns=["商品标题", "售价(分)"],
        rows=[{"商品标题": "甲", "售价(分)": "1990"}],
    )
    mapping = ColumnMapping.auto(table.columns)
    assert mapping.price_unit == "fen"
    product = build_products(table, mapping).products[0]
    assert product.price == pytest.approx(19.9)


def test_build_products_jin_weight_unit():
    table = TableData(
        columns=["商品标题", "成本", "重量(斤)"],
        rows=[{"商品标题": "甲", "成本": "10", "重量(斤)": "2"}],
    )
    mapping = ColumnMapping.auto(table.columns)
    assert mapping.weight_unit == "jin"
    product = build_products(table, mapping).products[0]
    assert product.weight_kg == pytest.approx(1.0)


def test_build_products_dedupes_by_external_id():
    table = TableData(
        columns=["商品标题", "商品ID", "成本"],
        rows=[
            {"商品标题": "甲", "商品ID": "X1", "成本": "10"},
            {"商品标题": "甲重复", "商品ID": "X1", "成本": "12"},
        ],
    )
    result = build_products(table, ColumnMapping.auto(table.columns))
    assert len(result.products) == 1
    assert result.skipped == 1


def test_build_products_skips_rows_without_price():
    table = TableData(
        columns=["商品标题", "成本"],
        rows=[{"商品标题": "有价", "成本": "10"}, {"商品标题": "无价", "成本": "暂无"}],
    )
    result = build_products(table, ColumnMapping.auto(table.columns))
    assert len(result.products) == 1
    assert result.skipped == 1
    assert any("缺少价格" in w for w in result.warnings)


def test_build_products_requires_title_mapping():
    table = TableData(columns=["成本", "销量"], rows=[{"成本": "10", "销量": "5"}])
    result = build_products(table, ColumnMapping.auto(table.columns))
    assert result.products == []
    assert any("商品标题" in w for w in result.warnings)


def test_build_products_empty_table():
    assert build_products(TableData([], []), ColumnMapping()).products == []


def test_build_products_warns_about_missing_mapped_column():
    table = TableData(columns=["商品标题", "成本"], rows=[{"商品标题": "甲", "成本": "10"}])
    mapping = ColumnMapping(fields={"title": "商品标题", "cost": "成本", "sales": "销量"})
    result = build_products(table, mapping)
    assert any("不存在" in w for w in result.warnings)


def test_build_products_defaults_for_unmapped_dimensions(ali_csv):
    data = read_csv(ali_csv)
    result = build_products(data, ColumnMapping.auto(data.columns))
    product = result.products[0]
    assert product.competition == 50.0
    assert product.compliance_risk == 20.0
    assert product.virality == 50.0


def test_build_products_products_validate():
    table = TableData(
        columns=["商品标题", "成本", "重量(g)"],
        rows=[{"商品标题": "甲", "成本": "10", "重量(g)": "500"}],
    )
    product = build_products(table, ColumnMapping.auto(table.columns)).products[0]
    assert isinstance(product, ProductIn)
    assert 0 <= product.heat <= 100
    assert 0 < product.weight_kg <= 50


# --------------------------------------------------------------------------- #
# 来源说明
# --------------------------------------------------------------------------- #

def test_provenance_marks_derived_fields(ali_csv):
    data = read_csv(ali_csv)
    mapping = ColumnMapping.auto(data.columns)
    result = build_products(data, mapping)
    text = result.provenance
    assert "表格导入" in text
    assert "售价 = 采购价" in text          # 推导
    assert "重量按克换算" in text
    assert "热度" not in text.split("表中未提供")[-1], "热度由销量推导，不该列为未提供"


def test_provenance_reports_unused_columns():
    table = TableData(
        columns=["商品标题", "成本", "无关列A", "无关列B"],
        rows=[{"商品标题": "甲", "成本": "10", "无关列A": "1", "无关列B": "2"}],
    )
    text = build_provenance(
        ColumnMapping.auto(table.columns), table.columns, [], []
    )
    assert "表内未使用列" in text and "无关列A" in text


# --------------------------------------------------------------------------- #
# 数据源与映射复用
# --------------------------------------------------------------------------- #

def test_tabular_source_uses_filename_as_source_label(ali_csv):
    source = TabularSource(ali_csv)
    assert source.source_label == "1688导出"
    products = source.fetch()
    assert all(product.source == "1688导出" for product in products)


def test_tabular_source_appends_provenance_to_note(ali_csv):
    products = TabularSource(ali_csv).fetch()
    assert "｜数据说明：" in products[0].note
    assert "表格导入" in products[0].note


def test_tabular_source_accepts_mapping_dict(ali_csv):
    mapping = ColumnMapping(
        fields={"title": "商品标题", "cost": "批发价(元)"},
        markup=3.0, source="自定义",
    )
    products = TabularSource(ali_csv, mapping.to_dict()).fetch()
    assert len(products) == 3
    assert products[0].price == pytest.approx(18.5 * 3.0)
    assert products[0].source == "自定义"


def test_tabular_source_caches_loaded_table(ali_csv):
    source = TabularSource(ali_csv)
    assert source.load() is source.load()


def test_mapping_round_trip():
    original = ColumnMapping(fields={"title": "标题"}, price_unit="fen",
                             weight_unit="g", markup=1.8, source="x",
                             default_category="杂货")
    restored = ColumnMapping.from_dict(original.to_dict())
    assert restored == original
    assert ColumnMapping.from_dict({}).markup == 2.5


def test_mapping_describe_and_unmapped():
    mapping = ColumnMapping(fields={"title": "标题", "cost": "成本"})
    assert "title←标题" in mapping.describe()
    assert "sales" in mapping.unmapped()
    assert ColumnMapping().describe() == "（未映射任何列）"


# --------------------------------------------------------------------------- #
# 落库与方案复用
# --------------------------------------------------------------------------- #

def test_import_table_end_to_end(tmp_path, ali_csv):
    db_path = tmp_path / "t.db"
    db.init_db(db_path)

    report = service.import_table(ali_csv, save_as="1688导出模板", db_path=db_path)
    assert report["saved"] == 3
    assert report["skipped"] == 1
    assert "gb18030" in report["table"]
    assert report["mapping"]["source"] == "1688导出"

    stored = db.list_products(db_path=db_path)
    assert len(stored) == 3
    assert stored[0].price > 0
    assert "｜数据说明：" in stored[0].note


def test_mapping_profile_persisted_and_reused(tmp_path, ali_csv):
    path = tmp_path / "m.db"
    db.init_db(path)
    data = read_csv(ali_csv)
    mapping = ColumnMapping.auto(data.columns)
    fingerprint = column_fingerprint(data.columns)

    db.upsert_mapping_profile("1688", fingerprint, mapping.to_dict(), data.columns, path)
    found = db.find_mapping_by_fingerprint(fingerprint, path)
    assert found is not None
    assert found["mapping"]["fields"]["title"] == "商品标题"
    assert found["columns"] == data.columns

    restored = ColumnMapping.from_dict(found["mapping"])
    assert restored.fields == mapping.fields

    assert db.find_mapping_by_fingerprint("deadbeef", path) is None
    assert db.get_mapping_profile("1688", path)["name"] == "1688"
    assert db.delete_mapping_profile("1688", path) is True
    assert db.delete_mapping_profile("1688", path) is False


def test_mapping_profile_upsert_updates_in_place(tmp_path):
    path = tmp_path / "m.db"
    db.init_db(path)
    first = db.upsert_mapping_profile("方案", "fp1", {"fields": {"title": "A"}}, ["A"], path)
    second = db.upsert_mapping_profile("方案", "fp2", {"fields": {"title": "B"}}, ["B"], path)
    assert first["id"] == second["id"]
    assert second["mapping"]["fields"]["title"] == "B"
    assert len(db.list_mapping_profiles(path)) == 1


def test_profile_lookup_prefers_name_over_id(tmp_path):
    """方案名很可能是纯数字（如「1688」），不能因为 isdigit() 就当成 id 查。"""
    path = tmp_path / "n.db"
    db.init_db(path)

    db.upsert_mapping_profile("1688", "fp", {"fields": {"title": "商品标题"}}, ["商品标题"], path)
    record = db.get_mapping_profile("1688", path)
    assert record is not None, "名字是纯数字时也必须能按名字查到"
    assert record["name"] == "1688"
    assert db.delete_mapping_profile("1688", path) is True

    db.upsert_profile("1688", {"margin": 1.0}, "数字名权重方案", path)
    weight_record = db.get_profile("1688", path)
    assert weight_record is not None
    assert weight_record["name"] == "1688"
    assert db.delete_profile("1688", path) is True


def test_profile_lookup_still_supports_integer_id(tmp_path):
    path = tmp_path / "i.db"
    db.init_db(path)
    saved = db.upsert_mapping_profile("普通名", "fp", {"fields": {}}, [], path)
    assert db.get_mapping_profile(saved["id"], path)["name"] == "普通名"
    assert db.get_mapping_profile("999999", path) is None
