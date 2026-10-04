"""业务编排层测试。

``service.py`` 把「存储 + 规则打分 + 大模型」串起来，此前完全没有测试。
这里盯住的是编排的**分支**，不是打分算法本身（后者归 ``test_scoring.py``）：

1. LLM 开启 / 关闭 / 调用失败三条路径都要走通，且失败不能中断主流程
2. ``score_product`` 对不存在的商品返回 ``None``，而不是抛异常
3. 榜单在「已有打分」与「尚未打分」时走不同分支，后者不应偷偷落库
4. 权重解析优先级：显式 weights > 库内方案 > 内置预设 > 全局默认
"""

from __future__ import annotations

import pytest

from app import db, llm, service
from app.models import ProductIn
from app.scoring import MAX_LLM_ADJUSTMENT


@pytest.fixture(autouse=True)
def _llm_off(monkeypatch):
    """默认关掉大模型，避免测试依赖本机 .env 里的凭据。"""
    monkeypatch.setattr(llm, "is_available", lambda: False)


def make_product(title: str = "测试商品", *, source: str = "manual",
                 price: float = 100.0, cost: float = 40.0, **kwargs) -> ProductIn:
    return ProductIn(title=title, source=source, price=price, cost=cost, **kwargs)


def seed(db_path, titles=("A", "B", "C", "D")):
    """写入若干带差异的商品，便于产生可区分的排名。"""
    products = []
    for index, title in enumerate(titles):
        products.append(db.upsert_product(
            make_product(
                title,
                price=100.0 + index * 20,
                cost=30.0 + index * 5,
                heat=90.0 - index * 15,
                competition=20.0 + index * 15,
                virality=80.0 - index * 10,
            ),
            db_path,
        ))
    return products


# --------------------------------------------------------------------------- #
# 商品导入与打分
# --------------------------------------------------------------------------- #

def test_import_products_dedupes(temp_db):
    first = service.import_products([make_product("保温杯", price=100.0)])
    second = service.import_products([make_product("保温杯", price=120.0)])

    assert first[0].id == second[0].id
    assert second[0].price == 120.0
    assert len(db.list_products(db_path=temp_db)) == 1


def test_score_product_missing_returns_none(temp_db):
    assert service.score_product(999) is None


def test_score_product_returns_full_breakdown_and_persists(temp_db):
    product = db.upsert_product(make_product("保温杯"), temp_db)

    result = service.score_product(product.id, use_llm=False)

    assert result["product_id"] == product.id
    assert 0.0 <= result["total"] <= 100.0
    assert result["grade"] in {"S", "A", "B", "C", "D"}
    assert set(result["dimensions"]) == {
        "demand", "competition", "margin", "shipping",
        "repurchase", "compliance", "virality",
    }
    assert result["llm_review"] is None
    assert result["llm_adjustment"] == 0.0
    assert result["advice"]

    assert len(db.latest_scores(db_path=temp_db)) == 1


def test_score_product_persist_false_does_not_write(temp_db):
    product = db.upsert_product(make_product("保温杯"), temp_db)
    service.score_product(product.id, use_llm=False, persist=False)
    assert db.latest_scores(db_path=temp_db) == []


def test_score_product_applies_llm_adjustment_and_grade(temp_db, monkeypatch):
    product = db.upsert_product(make_product("保温杯"), temp_db)
    baseline = service.score_product(product.id, use_llm=False, persist=False)

    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "review_product", lambda *a, **k: ("值得主推", 5.0))

    got = service.score_product(product.id, use_llm=True)

    assert got["llm_review"] == "值得主推"
    assert got["llm_adjustment"] == 5.0
    assert got["total"] == round(min(100.0, baseline["total"] + 5.0), 2)


def test_score_product_llm_failure_degrades_silently(temp_db, monkeypatch):
    """LLM 返回空点评时不得报错，也不得改动分数。"""
    product = db.upsert_product(make_product("保温杯"), temp_db)
    baseline = service.score_product(product.id, use_llm=False, persist=False)

    monkeypatch.setattr(llm, "is_available", lambda: True)
    monkeypatch.setattr(llm, "review_product", lambda *a, **k: (None, 0.0))

    got = service.score_product(product.id, use_llm=True)

    assert got["llm_review"] is None
    assert got["total"] == baseline["total"]


def test_review_product_clamps_adjustment_to_limit(monkeypatch):
    """模型吐 999 分修正时，必须被截断在 ±MAX_LLM_ADJUSTMENT。"""
    product = ProductIn(title="保温杯")
    monkeypatch.setattr(llm, "chat", lambda *a, **k: '{"review": "很值得", "adjustment": 999}')
    review, adjustment = llm.review_product(product, {"demand": 50.0}, 70.0)
    assert review == "很值得"
    assert adjustment == MAX_LLM_ADJUSTMENT

    monkeypatch.setattr(llm, "chat", lambda *a, **k: '{"review": "差", "adjustment": -999}')
    _, adjustment = llm.review_product(product, {"demand": 50.0}, 70.0)
    assert adjustment == -MAX_LLM_ADJUSTMENT


def test_review_product_handles_non_json_output(monkeypatch):
    """模型没按 JSON 返回时，原文当点评、不做修正。"""
    product = ProductIn(title="保温杯")
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "我觉得这个不错")
    review, adjustment = llm.review_product(product, {"demand": 50.0}, 70.0)
    assert review == "我觉得这个不错"
    assert adjustment == 0.0

    monkeypatch.setattr(llm, "chat", lambda *a, **k: None)
    assert llm.review_product(product, {"demand": 50.0}, 70.0) == (None, 0.0)


def test_score_all_returns_descending_totals(temp_db):
    seed(temp_db)
    results = service.score_all(use_llm=False)

    assert len(results) == 4
    totals = [item["total"] for item in results]
    assert totals == sorted(totals, reverse=True)


# --------------------------------------------------------------------------- #
# 榜单
# --------------------------------------------------------------------------- #

def test_leaderboard_computes_on_the_fly_without_persisting(temp_db):
    db.upsert_product(make_product("A"), temp_db)

    board = service.leaderboard()

    assert len(board) == 1
    assert board[0]["product_id"]
    assert db.latest_scores(db_path=temp_db) == []  # 未打分不应偷偷落库


def test_leaderboard_prefers_persisted_scores(temp_db):
    seed(temp_db)
    service.score_all(use_llm=False)

    board = service.leaderboard(limit=2)

    assert len(board) == 2
    assert board[0]["total"] >= board[1]["total"]
    assert "title" in board[0]  # 走的是 scores 表联查的分支


def test_leaderboard_filters_by_category(temp_db):
    db.upsert_product(make_product("A", category="家居"), temp_db)
    db.upsert_product(make_product("B", category="数码"), temp_db)

    assert [item["title"] for item in service.leaderboard(category="数码")] == ["B"]


def test_leaderboard_filter_with_no_match_returns_empty(temp_db):
    db.upsert_product(make_product("A", category="家居"), temp_db)
    assert service.leaderboard(category="不存在") == []


def test_dashboard_stats(temp_db):
    seed(temp_db, titles=("A", "B"))
    service.score_all(use_llm=False)

    stats = service.dashboard_stats()

    assert stats["total"] == 2
    assert stats["scored"] == 2
    assert stats["avg_score"] > 0


# --------------------------------------------------------------------------- #
# 权重方案
# --------------------------------------------------------------------------- #

def test_save_profile_rejects_invalid_weights(temp_db):
    with pytest.raises(ValueError):
        service.save_profile("bad", {"nope": 1.0})
    with pytest.raises(ValueError):
        service.save_profile("zero", {"demand": 0.0})


def test_save_profile_normalizes_weights(temp_db):
    saved = service.save_profile("毛利优先", {"margin": 3.0, "demand": 1.0})
    assert saved["weights"]["margin"] == 0.75
    assert saved["weights"]["demand"] == 0.25


def test_install_presets_is_idempotent(temp_db):
    assert service.install_presets() == 4
    assert service.install_presets() == 4
    assert len(db.list_profiles(db_path=temp_db)) == 4


# --------------------------------------------------------------------------- #
# 快照与对比
# --------------------------------------------------------------------------- #

def test_create_snapshot_requires_products(temp_db):
    with pytest.raises(ValueError):
        service.create_snapshot("空库")


def test_create_snapshot_with_explicit_weights(temp_db):
    seed(temp_db)

    run = service.create_snapshot("均衡", weights={"margin": 0.5, "demand": 0.5})

    assert run["product_count"] == 4
    assert run["avg_score"] > 0
    assert run["weights"]["margin"] == 0.5
    assert run["weights"]["demand"] == 0.5
    assert run["profile_id"] is None


def test_create_snapshot_with_saved_profile(temp_db):
    seed(temp_db)
    service.save_profile("毛利优先", {"margin": 0.8, "demand": 0.2})

    run = service.create_snapshot("用方案", profile="毛利优先")

    assert run["profile_id"] is not None
    assert sum(run["weights"].values()) == pytest.approx(1.0)


def test_create_snapshot_with_builtin_preset(temp_db):
    seed(temp_db)
    run = service.create_snapshot("用预设", profile="balanced")
    assert sum(run["weights"].values()) == pytest.approx(1.0)


def test_create_snapshot_rejects_unknown_profile(temp_db):
    seed(temp_db)
    with pytest.raises(KeyError):
        service.create_snapshot("x", profile="不存在")


def test_create_snapshot_rejects_invalid_weights(temp_db):
    seed(temp_db)
    with pytest.raises(ValueError):
        service.create_snapshot("x", weights={"nope": 1.0})


def test_snapshot_detail_and_missing(temp_db):
    seed(temp_db)
    run = service.create_snapshot("均衡", profile="balanced")

    detail = service.snapshot_detail(run["id"])
    assert len(detail["items"]) == 4
    assert [item["rank_no"] for item in detail["items"]] == [1, 2, 3, 4]

    with pytest.raises(KeyError):
        service.snapshot_detail(404)


def test_compare_snapshots_across_weights(temp_db):
    seed(temp_db, titles=("A", "B", "C", "D", "E", "F"))
    left = service.create_snapshot("均衡", profile="balanced")
    right = service.create_snapshot("毛利", profile="margin_first")

    result = service.compare_snapshots(left["id"], right["id"])

    assert result.common == 6
    assert result.summary()
    assert isinstance(result.spearman, float)


def test_compare_snapshots_missing_raises(temp_db):
    seed(temp_db)
    run = service.create_snapshot("均衡", profile="balanced")
    with pytest.raises(KeyError):
        service.compare_snapshots(run["id"], 404)
    with pytest.raises(KeyError):
        service.compare_snapshots(404, run["id"])


def test_snapshot_sensitivity(temp_db):
    seed(temp_db, titles=("A", "B", "C", "D", "E", "F"))
    run = service.create_snapshot("均衡", profile="balanced")

    impacts = service.snapshot_sensitivity(run["id"])

    assert isinstance(impacts, list)
    assert impacts
    assert {impact.dimension for impact in impacts} <= set(
        {"demand", "competition", "margin", "shipping",
         "repurchase", "compliance", "virality"}
    )

    with pytest.raises(KeyError):
        service.snapshot_sensitivity(404)


def test_list_snapshots(temp_db):
    seed(temp_db)
    service.create_snapshot("第一次", profile="balanced")
    service.create_snapshot("第二次", profile="balanced")
    assert [run["label"] for run in service.list_snapshots()] == ["第二次", "第一次"]


# --------------------------------------------------------------------------- #
# 经营结果与回测
# --------------------------------------------------------------------------- #

def test_record_outcome_validates_product(temp_db):
    with pytest.raises(KeyError):
        service.record_outcome(999, "2024-03-01", "2024-03-31", orders=1)


def test_record_outcome_validates_dates(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)

    with pytest.raises(ValueError):
        service.record_outcome(product.id, "2024/13/01", "2024-03-31")
    with pytest.raises(ValueError):
        service.record_outcome(product.id, "2024-04-01", "2024-03-31")


def test_record_outcome_rejects_negative_values(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    with pytest.raises(ValueError):
        service.record_outcome(product.id, "2024-03-01", "2024-03-31", revenue=-1)


def test_record_outcome_normalises_date_format(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    saved = service.record_outcome(product.id, "2024/3/1", "2024/3/31", orders=5)
    assert saved["window_start"] == "2024-03-01"
    assert saved["window_end"] == "2024-03-31"


def test_outcome_listing_and_deletion(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)
    row = service.record_outcome(product.id, "2024-03-01", "2024-03-31", orders=5)

    assert len(service.list_outcomes()) == 1
    assert service.list_outcomes(product_id=product.id)[0]["orders"] == 5.0
    assert service.delete_outcome(row["id"]) is True
    assert service.list_outcomes() == []


def test_record_decision_validations(temp_db):
    product = db.upsert_product(make_product("A"), temp_db)

    with pytest.raises(KeyError):
        service.record_decision(999, action="push")
    with pytest.raises(ValueError):
        service.record_decision(product.id, action="乱写")
    with pytest.raises(KeyError):
        service.record_decision(product.id, run_id=999, action="push")

    decision = service.record_decision(product.id, action="push", note="主推")
    assert decision["action"] == "push"
    assert service.list_decisions(product_id=product.id)[0]["note"] == "主推"


def test_backtest_requires_existing_run(temp_db):
    with pytest.raises(KeyError):
        service.backtest_run(999)


def test_backtest_rejects_unknown_metric(temp_db):
    seed(temp_db)
    run = service.create_snapshot("均衡", profile="balanced")
    with pytest.raises(ValueError):
        service.backtest_run(run["id"], metric="不存在")


def test_backtest_end_to_end(temp_db):
    products = seed(temp_db, titles=("A", "B", "C", "D", "E", "F"))
    run = service.create_snapshot("均衡", profile="balanced")
    for index, product in enumerate(products):
        service.record_outcome(product.id, "2024-03-01", "2024-03-31",
                               revenue=1000.0 - index * 100, cogs=0.0)

    result = service.backtest_run(run["id"], metric="gross_profit")

    assert result.sample_size == 6
    assert result.rho is not None
    assert result.summary()


def test_compare_backtests_end_to_end(temp_db):
    products = seed(temp_db, titles=("A", "B", "C", "D", "E", "F"))
    left = service.create_snapshot("均衡", profile="balanced")
    right = service.create_snapshot("毛利", profile="margin_first")
    for index, product in enumerate(products):
        service.record_outcome(product.id, "2024-03-01", "2024-03-31",
                               revenue=1000.0 - index * 100, cogs=0.0)

    comparison = service.compare_backtests(left["id"], right["id"], metric="gross_profit")

    assert comparison.better in {"left", "right", "tie"}
    assert comparison.summary()


def test_backtest_after_run_only_excludes_old_windows(temp_db):
    """快照是现在创建的，2020 年的结果明显早于它 —— 不该参与回测。"""
    products = seed(temp_db, titles=("A", "B", "C", "D", "E", "F"))
    run = service.create_snapshot("均衡", profile="balanced")
    for product in products:
        service.record_outcome(product.id, "2020-01-01", "2020-01-31", revenue=100.0)

    loose = service.backtest_run(run["id"], metric="gross_profit")
    strict = service.backtest_run(run["id"], metric="gross_profit", after_run_only=True)

    assert loose.sample_size == 6
    assert any("早于快照" in note for note in loose.notes)
    assert strict.sample_size == 0
    assert strict.excluded_before_run == 6
