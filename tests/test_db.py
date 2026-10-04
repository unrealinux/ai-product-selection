"""SQLite 存储层测试。

``db.py`` 是数据正确性的必经之路，此前完全没有测试。这里盯住五类会静默出错的地方：

1. **去重语义** —— 同 ``(title, source)`` 应更新而非新增，否则重复导入会翻倍
2. **「取最新一条打分」** —— 子查询写错会静默返回旧分数，界面上看不出来
3. **外键级联** —— 删快照必须带走条目，否则监听数据里留下孤儿行
4. **旧库自动补列** —— 迁移必须幂等，且不能丢已有数据
5. **方案查找优先级** —— 方案名可能是纯数字（「1688」），先按名字再按 id
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from app import db
from app.models import ProductIn


# --------------------------------------------------------------------------- #
# 造数据
# --------------------------------------------------------------------------- #

def make_product(title: str = "测试商品", *, source: str = "manual",
                 price: float = 100.0, cost: float = 40.0, **kwargs) -> ProductIn:
    return ProductIn(title=title, source=source, price=price, cost=cost, **kwargs)


def make_score(product_id: int, *, total: float = 80.0, grade: str = "A", **kwargs) -> dict:
    payload = {
        "product_id": product_id,
        "total": total,
        "grade": grade,
        "dimensions": {"demand": 90.0, "margin": 70.0},
        "profit_margin": 0.6,
        "advice": "建议",
        "llm_review": None,
        "llm_adjustment": 0.0,
        "scored_at": datetime(2024, 1, 1, 12, 0, 0),
    }
    payload.update(kwargs)
    return payload


# --------------------------------------------------------------------------- #
# 建表与迁移
# --------------------------------------------------------------------------- #

def test_init_db_creates_all_tables(temp_db):
    db.init_db(temp_db)  # 第二次调用必须幂等
    with db.session(temp_db) as conn:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {
        "products", "scores", "weight_profiles",
        "score_runs", "score_run_items", "mapping_profiles",
    } <= names


def test_migration_backfills_missing_column_without_data_loss(tmp_path):
    """旧库缺 external_id 时自动补列，已有行必须保留。"""
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "title TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    raw.execute(
        "INSERT INTO products (title, source, created_at) VALUES (?, ?, ?)",
        ("旧商品", "manual", "2024-01-01T00:00:00"),
    )
    raw.commit()
    raw.close()

    db.init_db(path)

    with db.session(path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(products)")}
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert "external_id" in columns
    assert "image_url" in columns
    assert count == 1


def test_session_rolls_back_on_error(temp_db):
    """同一事务里第二句失败，第一句也必须回滚。"""
    insert = "INSERT INTO products (title, source, created_at) VALUES ('x', 'y', 'z')"
    with pytest.raises(sqlite3.IntegrityError):
        with db.session(temp_db) as conn:
            conn.execute(insert)
            conn.execute(insert)  # UNIQUE (title, source) 冲突

    with db.session(temp_db) as conn:
        left = conn.execute("SELECT COUNT(*) FROM products WHERE title = 'x'").fetchone()[0]
    assert left == 0


# --------------------------------------------------------------------------- #
# 商品写入与查询
# --------------------------------------------------------------------------- #

def test_upsert_dedupes_by_title_and_source(temp_db):
    first = db.upsert_product(make_product("保温杯", price=100.0), temp_db)
    second = db.upsert_product(make_product("保温杯", price=150.0), temp_db)

    assert first.id == second.id
    assert second.price == 150.0
    assert len(db.list_products(db_path=temp_db)) == 1


def test_upsert_keeps_same_title_from_different_sources(temp_db):
    db.upsert_product(make_product("保温杯", source="1688"), temp_db)
    db.upsert_product(make_product("保温杯", source="taobao"), temp_db)
    assert len(db.list_products(db_path=temp_db)) == 2


def test_get_product_missing_returns_none(temp_db):
    assert db.get_product(999, temp_db) is None


def test_list_products_filters_and_limits(temp_db):
    db.upsert_product(make_product("A", category="家居", source="1688"), temp_db)
    db.upsert_product(make_product("B", category="数码", source="1688"), temp_db)
    db.upsert_product(make_product("C", category="数码", source="taobao"), temp_db)

    assert {p.title for p in db.list_products(category="数码", db_path=temp_db)} == {"B", "C"}
    assert {p.title for p in db.list_products(source="1688", db_path=temp_db)} == {"A", "B"}
    assert len(db.list_products(source="1688", limit=1, db_path=temp_db)) == 1
    assert len(db.list_products(db_path=temp_db)) == 3


def test_bulk_upsert_returns_saved_records(temp_db):
    saved = db.bulk_upsert([make_product("A"), make_product("B")], temp_db)
    assert [p.title for p in saved] == ["A", "B"]
    assert all(p.id for p in saved)


def test_image_url_round_trips(temp_db):
    saved = db.upsert_product(
        make_product("A", image_url="https://img.alicdn.com/a.jpg"), temp_db
    )
    assert db.get_product(saved.id, temp_db).image_url == "https://img.alicdn.com/a.jpg"


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #

def test_latest_scores_returns_only_newest_row_per_product(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    db.save_score(make_score(product.id, total=50.0, grade="D"), temp_db)
    db.save_score(make_score(product.id, total=90.0, grade="S"), temp_db)

    rows = db.latest_scores(db_path=temp_db)

    assert len(rows) == 1
    assert rows[0]["total"] == 90.0
    assert rows[0]["grade"] == "S"
    assert rows[0]["title"] == "A"
    assert rows[0]["dimensions"] == {"demand": 90.0, "margin": 70.0}
    assert isinstance(rows[0]["scored_at"], datetime)


def test_latest_scores_orders_by_total_desc(temp_db):
    low = db.upsert_product(make_product("低分"), temp_db)
    high = db.upsert_product(make_product("高分"), temp_db)
    db.save_score(make_score(low.id, total=30.0, grade="D"), temp_db)
    db.save_score(make_score(high.id, total=88.0, grade="S"), temp_db)

    titles = [row["title"] for row in db.latest_scores(db_path=temp_db)]
    assert titles == ["高分", "低分"]


def test_latest_score_for_single_product(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    assert db.latest_score(product.id, temp_db) is None

    db.save_score(make_score(product.id, total=71.0, grade="B"), temp_db)
    assert db.latest_score(product.id, temp_db)["total"] == 71.0
    assert db.latest_score(999, temp_db) is None


def test_save_score_accepts_string_and_default_timestamp(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)

    db.save_score(make_score(product.id, scored_at="2024-05-05T10:00:00"), temp_db)
    without = make_score(product.id, total=60.0)
    without.pop("scored_at")  # 走默认分支
    db.save_score(without, temp_db)

    rows = db.latest_scores(db_path=temp_db)
    assert len(rows) == 1  # 同一商品只留最新一条
    assert rows[0]["total"] == 60.0
    assert isinstance(rows[0]["scored_at"], datetime)


def test_stats_reports_latest_scores_only(temp_db):
    a = db.upsert_product(make_product("A", category="家居"), temp_db)
    b = db.upsert_product(make_product("B", category="数码"), temp_db)
    db.save_score(make_score(a.id, total=90.0, grade="S"), temp_db)
    db.save_score(make_score(b.id, total=70.0, grade="B"), temp_db)
    db.save_score(make_score(b.id, total=20.0, grade="D"), temp_db)  # 旧分不应计入

    result = db.stats(temp_db)

    assert result["total"] == 2
    assert result["scored"] == 2
    # A 的 90 与 B 的最新一条 20，B 的旧分 70 不应计入
    assert result["avg_score"] == 55.0
    assert result["grade_distribution"] == {"S": 1, "D": 1}
    assert [c["category"] for c in result["top_categories"]] == ["家居", "数码"]


def test_stats_on_empty_db_is_zeroed(temp_db):
    result = db.stats(temp_db)
    assert result["total"] == 0
    assert result["avg_score"] == 0.0
    assert result["grade_distribution"] == {}


# --------------------------------------------------------------------------- #
# 权重方案
# --------------------------------------------------------------------------- #

def test_upsert_profile_updates_in_place(temp_db):
    created = db.upsert_profile("均衡", {"demand": 1.0}, "初版", temp_db)
    updated = db.upsert_profile("均衡", {"demand": 0.5, "margin": 0.5}, "改过", temp_db)

    assert created["id"] == updated["id"]
    assert updated["weights"] == {"demand": 0.5, "margin": 0.5}
    assert updated["description"] == "改过"
    assert len(db.list_profiles(temp_db)) == 1


def test_profile_lookup_prefers_name_over_numeric_id(temp_db):
    """方案名可能是「1688」这种纯数字，先查名字才不会静默查错。"""
    db.upsert_profile("其他", {"demand": 1.0}, db_path=temp_db)  # id = 1
    db.upsert_profile("1", {"margin": 1.0}, db_path=temp_db)     # id = 2，名字是 "1"

    found = db.get_profile(1, temp_db)

    assert found is not None
    assert found["name"] == "1"


def test_profile_lookup_by_integer_id_when_name_not_found(temp_db):
    db.upsert_profile("均衡", {"demand": 1.0}, db_path=temp_db)
    assert db.get_profile(1, temp_db)["name"] == "均衡"


def test_delete_profile(temp_db):
    db.upsert_profile("均衡", {"demand": 1.0}, db_path=temp_db)
    assert db.delete_profile("均衡", temp_db) is True
    assert db.delete_profile("均衡", temp_db) is False
    assert db.list_profiles(temp_db) == []


# --------------------------------------------------------------------------- #
# 打分快照
# --------------------------------------------------------------------------- #

def test_create_run_ranks_items_and_cascades_on_delete(temp_db):
    low = db.upsert_product(make_product("低分"), temp_db)
    high = db.upsert_product(make_product("高分"), temp_db)

    run = db.create_run(
        "baseline",
        {"demand": 1.0},
        [
            {"product_id": low.id, "total": 60.0, "title": "低分",
             "dimensions": {"demand": 60.0}},
            {"product_id": high.id, "total": 90.0, "title": "高分",
             "dimensions": {"demand": 90.0}},
        ],
        db_path=temp_db,
    )

    assert run["product_count"] == 2
    assert run["avg_score"] == 75.0
    assert run["weights"] == {"demand": 1.0}

    items = db.get_run_items(run["id"], temp_db)
    assert [item["rank_no"] for item in items] == [1, 2]
    assert items[0]["title"] == "高分"
    assert items[0]["dimensions"] == {"demand": 90.0}

    assert db.delete_run(run["id"], temp_db) is True
    assert db.get_run(run["id"], temp_db) is None
    assert db.get_run_items(run["id"], temp_db) == []  # 级联删除


def test_create_run_without_items_does_not_divide_by_zero(temp_db):
    run = db.create_run("空", {"demand": 1.0}, [], db_path=temp_db)
    assert run["product_count"] == 0
    assert run["avg_score"] == 0.0


def test_list_runs_is_newest_first(temp_db):
    db.create_run("第一次", {"demand": 1.0}, [], db_path=temp_db)
    db.create_run("第二次", {"demand": 1.0}, [], db_path=temp_db)
    assert [run["label"] for run in db.list_runs(db_path=temp_db)] == ["第二次", "第一次"]


def test_delete_missing_run_returns_false(temp_db):
    assert db.delete_run(404, temp_db) is False


# --------------------------------------------------------------------------- #
# 表格映射方案
# --------------------------------------------------------------------------- #

def test_mapping_profile_round_trip(temp_db):
    db.upsert_mapping_profile(
        "1688", "fp-1", {"title": "商品标题"}, ["商品标题", "供货价"], db_path=temp_db
    )

    found = db.find_mapping_by_fingerprint("fp-1", temp_db)
    assert found is not None
    assert found["mapping"] == {"title": "商品标题"}
    assert found["columns"] == ["商品标题", "供货价"]


def test_find_mapping_by_missing_fingerprint_returns_none(temp_db):
    assert db.find_mapping_by_fingerprint("不存在", temp_db) is None


def test_mapping_profile_lookup_prefers_name(temp_db):
    db.upsert_mapping_profile("其他", "fp-a", {}, db_path=temp_db)  # id = 1
    db.upsert_mapping_profile("2", "fp-b", {}, db_path=temp_db)     # id = 2，名字 "2"

    assert db.get_mapping_profile(2, temp_db)["name"] == "2"


def test_upsert_mapping_profile_updates_in_place(temp_db):
    first = db.upsert_mapping_profile("1688", "fp-1", {"title": "A"}, ["A"], db_path=temp_db)
    second = db.upsert_mapping_profile("1688", "fp-2", {"title": "B"}, ["B"], db_path=temp_db)

    assert first["id"] == second["id"]
    assert second["fingerprint"] == "fp-2"
    assert second["columns"] == ["B"]


def test_delete_mapping_profile(temp_db):
    db.upsert_mapping_profile("1688", "fp-1", {}, db_path=temp_db)
    assert db.delete_mapping_profile("1688", temp_db) is True
    assert db.delete_mapping_profile("1688", temp_db) is False
    assert db.list_mapping_profiles(temp_db) == []


# --------------------------------------------------------------------------- #
# 经营结果（回测用）
# --------------------------------------------------------------------------- #

def test_upsert_outcome_updates_same_window_in_place(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)

    first = db.upsert_outcome(product.id, "2024-03-01", "2024-03-31",
                              orders=10, revenue=1000.0, db_path=temp_db)
    second = db.upsert_outcome(product.id, "2024-03-01", "2024-03-31",
                               orders=12, revenue=1500.0, db_path=temp_db)

    assert first["id"] == second["id"]  # 同一窗口 → 更新而非新增
    assert second["orders"] == 12.0
    assert second["revenue"] == 1500.0
    assert len(db.list_outcomes(db_path=temp_db)) == 1


def test_outcomes_keeps_distinct_windows(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    db.upsert_outcome(product.id, "2024-01-01", "2024-01-31", orders=1, db_path=temp_db)
    db.upsert_outcome(product.id, "2024-02-01", "2024-02-29", orders=2, db_path=temp_db)
    assert len(db.list_outcomes(db_path=temp_db)) == 2


def test_list_outcomes_filters_and_orders_desc(temp_db):
    a = db.upsert_product(make_product("A"), temp_db)
    b = db.upsert_product(make_product("B"), temp_db)
    db.upsert_outcome(a.id, "2024-01-01", "2024-01-31", orders=1, db_path=temp_db)
    db.upsert_outcome(a.id, "2024-03-01", "2024-03-31", orders=3, db_path=temp_db)
    db.upsert_outcome(b.id, "2024-02-01", "2024-02-29", orders=2, db_path=temp_db)

    only_a = db.list_outcomes(product_id=a.id, db_path=temp_db)

    assert len(only_a) == 2
    assert [row["window_start"] for row in only_a] == ["2024-03-01", "2024-01-01"]
    assert len(db.list_outcomes(db_path=temp_db)) == 3


def test_delete_outcome(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    row = db.upsert_outcome(product.id, "2024-01-01", "2024-01-31", db_path=temp_db)

    assert db.delete_outcome(row["id"], temp_db) is True
    assert db.delete_outcome(row["id"], temp_db) is False
    assert db.list_outcomes(db_path=temp_db) == []


def test_deleting_product_cascades_outcomes_and_decisions(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    db.upsert_outcome(product.id, "2024-01-01", "2024-01-31", orders=5, db_path=temp_db)
    db.record_decision(product.id, action="push", db_path=temp_db)

    with db.session(temp_db) as conn:
        conn.execute("DELETE FROM products WHERE id = ?", (product.id,))

    assert db.list_outcomes(db_path=temp_db) == []
    assert db.list_decisions(db_path=temp_db) == []


# --------------------------------------------------------------------------- #
# 决策记录
# --------------------------------------------------------------------------- #

def test_record_and_filter_decisions(temp_db):
    a = db.upsert_product(make_product("A"), temp_db)
    b = db.upsert_product(make_product("B"), temp_db)
    run = db.create_run("基线", {"demand": 1.0}, [], db_path=temp_db)

    db.record_decision(a.id, run_id=run["id"], action="push", note="主推", db_path=temp_db)
    db.record_decision(b.id, run_id=run["id"], action="skip", db_path=temp_db)
    db.record_decision(a.id, action="hold", db_path=temp_db)

    assert len(db.list_decisions(db_path=temp_db)) == 3
    assert [d["action"] for d in db.list_decisions(run_id=run["id"], db_path=temp_db)] == ["skip", "push"]
    assert len(db.list_decisions(product_id=a.id, db_path=temp_db)) == 2
    assert db.list_decisions(run_id=run["id"], db_path=temp_db)[1]["note"] == "主推"


def test_deleting_run_keeps_decision_but_nulls_run_id(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    run = db.create_run("基线", {"demand": 1.0}, [], db_path=temp_db)
    db.record_decision(product.id, run_id=run["id"], db_path=temp_db)

    db.delete_run(run["id"], temp_db)

    decisions = db.list_decisions(db_path=temp_db)
    assert len(decisions) == 1
    assert decisions[0]["run_id"] is None


def test_delete_decision(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    decision = db.record_decision(product.id, db_path=temp_db)

    assert db.delete_decision(decision["id"], temp_db) is True
    assert db.delete_decision(decision["id"], temp_db) is False
    assert db.list_decisions(db_path=temp_db) == []
