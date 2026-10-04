"""FastAPI 路由测试。

``api.py`` 有 22 个端点，此前没有任何测试。这里做两件事：

1. **冒烟**：每个端点至少被调用一次，确认路由、响应模型与序列化没问题
2. **可修复的错误映射成正确状态码**：未知数据源 → 400、文件不存在 → 404、
   校验失败 → 422、缺失但可修改的凭据 → 400（而不是 500）

所有请求都打在临时的 SQLite 上，不碰 ``data/products.db``。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client(temp_db):
    with TestClient(app) as test_client:
        yield test_client


def create(client, title: str, **kwargs):
    response = client.post("/products", json={"title": title, **kwargs})
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# 健康检查与商品 CRUD
# --------------------------------------------------------------------------- #

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_product_crud_round_trip(client):
    created = create(client, "保温杯", category="家居", source="1688", price=120.0, cost=45.0)
    assert created["id"]
    assert created["title"] == "保温杯"
    assert created["created_at"]

    listed = client.get("/products").json()
    assert [item["title"] for item in listed] == ["保温杯"]

    detail = client.get(f"/products/{created['id']}").json()
    assert detail["price"] == 120.0


def test_product_filters(client):
    create(client, "A", category="家居", source="1688")
    create(client, "B", category="数码", source="taobao")

    assert [p["title"] for p in client.get("/products", params={"category": "数码"}).json()] == ["B"]
    assert [p["title"] for p in client.get("/products", params={"source": "1688"}).json()] == ["A"]
    assert len(client.get("/products", params={"limit": 1}).json()) == 1


def test_missing_product_returns_404(client):
    assert client.get("/products/999").status_code == 404


def test_product_validation_errors(client):
    assert client.post("/products", json={}).status_code == 422
    assert client.post("/products", json={"title": "   "}).status_code == 422
    assert client.post("/products", json={"title": "x", "price": -1}).status_code == 422


def test_upsert_by_identical_title_updates_in_place(client):
    first = create(client, "保温杯", price=100.0)
    second = create(client, "保温杯", price=150.0)
    assert first["id"] == second["id"]
    assert len(client.get("/products").json()) == 1


# --------------------------------------------------------------------------- #
# 导入
# --------------------------------------------------------------------------- #

def test_import_sample_without_scoring(client):
    response = client.post("/import", json={"source": "sample", "score": False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["imported"] == 12
    assert body["scored"] == 0


def test_import_json_with_scoring(client, tmp_path):
    path = tmp_path / "products.json"
    path.write_text(json.dumps([
        {"title": "甲", "price": 100.0, "cost": 40.0, "heat": 90.0},
        {"title": "乙", "price": 200.0, "cost": 150.0, "heat": 20.0},
    ]), encoding="utf-8")

    response = client.post("/import", json={
        "source": "json", "path": str(path), "score": True, "use_llm": False,
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["imported"] == 2
    assert body["scored"] == 2
    totals = [item["total"] for item in body["items"]]
    assert totals == sorted(totals, reverse=True)


def test_import_unknown_source_returns_400(client):
    response = client.post("/import", json={"source": "火星源"})
    assert response.status_code == 400
    assert "未知数据源" in response.json()["detail"]


def test_import_missing_json_file_returns_404(client, tmp_path):
    response = client.post("/import", json={
        "source": "json", "path": str(tmp_path / "nope.json"),
    })
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# 打分 / 榜单 / 统计
# --------------------------------------------------------------------------- #

def test_score_single_product(client):
    product = create(client, "保温杯")
    body = client.post(f"/score/{product['id']}", params={"use_llm": False}).json()
    assert 0.0 <= body["total"] <= 100.0
    assert body["grade"] in {"S", "A", "B", "C", "D"}


def test_score_missing_product_returns_404(client):
    assert client.post("/score/999", params={"use_llm": False}).status_code == 404


def test_score_all(client):
    create(client, "A")
    create(client, "B")
    body = client.post("/score", params={"use_llm": False}).json()
    assert body["scored"] == 2
    assert len(body["items"]) == 2


def test_leaderboard_and_stats(client):
    create(client, "A", category="家居")
    create(client, "B", category="数码")

    board = client.get("/leaderboard").json()
    assert len(board) == 2
    assert board[0]["total"] >= board[1]["total"]

    # 榜单的兜底分支是即时算分、不落库
    stats = client.get("/stats").json()
    assert stats["total"] == 2
    assert stats["scored"] == 0

    client.post("/score", params={"use_llm": False})
    assert client.get("/stats").json()["scored"] == 2


def test_leaderboard_limit_validation(client):
    assert client.get("/leaderboard", params={"limit": 0}).status_code == 422


# --------------------------------------------------------------------------- #
# 权重方案
# --------------------------------------------------------------------------- #

def test_presets_endpoint(client):
    names = {item["name"] for item in client.get("/presets").json()}
    assert names == {"balanced", "margin_first", "traffic_first", "low_risk"}


def test_profile_crud(client):
    created = client.post("/profiles", json={"name": "my-plan", "weights": {"margin": 1.0}})
    assert created.status_code == 201, created.text

    assert [p["name"] for p in client.get("/profiles").json()] == ["my-plan"]

    deleted = client.delete("/profiles/my-plan")
    assert deleted.status_code == 200
    assert client.delete("/profiles/my-plan").status_code == 404


def test_profile_invalid_weights_returns_400(client):
    response = client.post("/profiles", json={"name": "bad", "weights": {"nope": 1.0}})
    assert response.status_code == 400


def test_install_presets_endpoint(client):
    assert client.post("/profiles/install-presets").json() == {"installed": 4}
    assert len(client.get("/profiles").json()) == 4


# --------------------------------------------------------------------------- #
# 快照与 A/B 对比
# --------------------------------------------------------------------------- #

def test_run_requires_products(client):
    response = client.post("/runs", json={"label": "空库", "profile": "balanced"})
    assert response.status_code == 400


def test_run_lifecycle(client):
    for index in range(4):
        create(client, f"商品{index}", heat=90.0 - index * 15, competition=20.0 + index * 15)

    created = client.post("/runs", json={"label": "均衡", "profile": "balanced"})
    assert created.status_code == 201, created.text
    run = created.json()
    assert run["product_count"] == 4
    assert run["avg_score"] > 0

    assert [r["id"] for r in client.get("/runs").json()] == [run["id"]]

    detail = client.get(f"/runs/{run['id']}").json()
    assert len(detail["items"]) == 4
    assert [item["rank_no"] for item in detail["items"]] == [1, 2, 3, 4]

    assert client.get("/runs/999").status_code == 404
    assert client.delete(f"/runs/{run['id']}").status_code == 200
    assert client.delete(f"/runs/{run['id']}").status_code == 404


def test_run_invalid_profile_returns_400(client):
    create(client, "商品")
    response = client.post("/runs", json={"label": "x", "profile": "不存在"})
    assert response.status_code == 400


def test_sensitivity_endpoint(client):
    for index in range(4):
        create(client, f"商品{index}", heat=90.0 - index * 15, competition=20.0 + index * 15)
    run = client.post("/runs", json={"label": "均衡", "profile": "balanced"}).json()

    body = client.get(f"/runs/{run['id']}/sensitivity").json()

    assert body["run_id"] == run["id"]
    assert body["impacts"]
    assert {"dimension", "influence", "verdict"} <= set(body["impacts"][0])

    assert client.get("/runs/999/sensitivity").status_code == 404


def test_compare_endpoint(client):
    for index in range(4):
        create(client, f"商品{index}", heat=90.0 - index * 15, competition=20.0 + index * 15)
    left = client.post("/runs", json={"label": "均衡", "profile": "balanced"}).json()
    right = client.post("/runs", json={"label": "毛利", "profile": "margin_first"}).json()

    body = client.get("/compare", params={"run_a": left["id"], "run_b": right["id"]}).json()

    assert body["common"] == 4
    assert body["summary"]
    assert body["run_a"]["id"] == left["id"]

    assert client.get("/compare", params={"run_a": left["id"], "run_b": 999}).status_code == 404


# --------------------------------------------------------------------------- #
# 抖音数据源
# --------------------------------------------------------------------------- #

def test_douyin_status(client):
    body = client.get("/douyin/status").json()
    assert "configured" in body
    assert body["error_codes"]


def test_douyin_token_without_credentials_returns_400(client, monkeypatch):
    import app.sources.douyin as douyin

    monkeypatch.setattr(douyin, "settings", SimpleNamespace(
        douyin_app_key="", douyin_app_secret="", douyin_access_token="",
        douyin_shop_id="", douyin_sign_method="hmac-sha256",
        douyin_base_url="http://127.0.0.1:1", douyin_timeout=1,
    ))

    response = client.post("/douyin/token", json={})

    assert response.status_code == 400
    assert "app_key" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 效果回测：经营结果 + 决策 + 回测
# --------------------------------------------------------------------------- #

OUTCOME_PAYLOAD = {
    "window_start": "2024-03-01",
    "window_end": "2024-03-31",
    "impressions": 1000,
    "clicks": 100,
    "orders": 20,
    "units": 25,
    "returns": 2,
    "revenue": 2000.0,
    "cogs": 800.0,
    "ad_spend": 200.0,
}


def test_metrics_endpoint(client):
    names = {item["name"] for item in client.get("/metrics").json()}
    assert {"gross_profit", "return_rate", "roi", "cvr"} <= names


def test_outcome_lifecycle(client):
    product = create(client, "保温杯")

    created = client.post("/outcomes", json={"product_id": product["id"], **OUTCOME_PAYLOAD})
    assert created.status_code == 201, created.text
    row = created.json()
    assert row["orders"] == 20.0

    listed = client.get("/outcomes", params={"product_id": product["id"]}).json()
    assert len(listed) == 1

    # 同一窗口重复提交 → 更新而非新增
    updated = client.post("/outcomes", json={
        "product_id": product["id"], **OUTCOME_PAYLOAD, "orders": 30,
    })
    assert updated.status_code == 201
    assert len(client.get("/outcomes").json()) == 1
    assert client.get("/outcomes").json()[0]["orders"] == 30.0

    assert client.delete(f"/outcomes/{row['id']}").status_code == 200
    assert client.delete(f"/outcomes/{row['id']}").status_code == 404


def test_outcome_unknown_product_returns_404(client):
    response = client.post("/outcomes", json={"product_id": 999, **OUTCOME_PAYLOAD})
    assert response.status_code == 404


def test_outcome_reversed_window_returns_400(client):
    product = create(client, "保温杯")
    response = client.post("/outcomes", json={
        "product_id": product["id"], **OUTCOME_PAYLOAD,
        "window_start": "2024-04-01", "window_end": "2024-03-31",
    })
    assert response.status_code == 400


def test_outcome_validation_errors(client):
    product = create(client, "保温杯")
    assert client.post("/outcomes", json={
        "product_id": product["id"], **OUTCOME_PAYLOAD, "window_start": "2024-13-99",
    }).status_code == 422
    assert client.post("/outcomes", json={
        "product_id": product["id"], **OUTCOME_PAYLOAD, "revenue": -1,
    }).status_code == 422


def test_decision_lifecycle(client):
    product = create(client, "保温杯")

    created = client.post("/decisions", json={"product_id": product["id"], "action": "push"})
    assert created.status_code == 201, created.text
    assert created.json()["action"] == "push"

    assert len(client.get("/decisions").json()) == 1
    assert len(client.get("/decisions", params={"product_id": product["id"]}).json()) == 1


def test_decision_errors(client):
    product = create(client, "保温杯")
    assert client.post("/decisions", json={"product_id": 999, "action": "push"}).status_code == 404
    assert client.post("/decisions", json={"product_id": product["id"], "action": "乱写"}).status_code == 422


def test_backtest_endpoint(client):
    ids = [create(client, f"商品{index}")["id"] for index in range(6)]
    run = client.post("/runs", json={"label": "均衡", "profile": "balanced"}).json()
    for index, product_id in enumerate(ids):
        client.post("/outcomes", json={
            "product_id": product_id, **OUTCOME_PAYLOAD,
            "revenue": 2000.0 - index * 200,
        })

    body = client.get(f"/runs/{run['id']}/backtest",
                      params={"metric": "gross_profit"}).json()

    assert body["sample_size"] == 6
    assert "summary" in body
    assert len(body["rows"]) == 6

    assert client.get("/runs/999/backtest").status_code == 404
    assert client.get(f"/runs/{run['id']}/backtest",
                      params={"metric": "不存在"}).status_code == 400
    assert client.get(f"/runs/{run['id']}/backtest",
                      params={"top_ratio": 0}).status_code == 422

    strict = client.get(f"/runs/{run['id']}/backtest",
                        params={"metric": "gross_profit", "after_run_only": True}).json()
    assert "excluded_before_run" in strict


def test_backtest_compare_endpoint(client):
    ids = [create(client, f"商品{index}")["id"] for index in range(6)]
    left = client.post("/runs", json={"label": "均衡", "profile": "balanced"}).json()
    right = client.post("/runs", json={"label": "毛利", "profile": "margin_first"}).json()
    for index, product_id in enumerate(ids):
        client.post("/outcomes", json={
            "product_id": product_id, **OUTCOME_PAYLOAD, "revenue": 2000.0 - index * 200,
        })

    body = client.get("/backtest/compare", params={
        "run_a": left["id"], "run_b": right["id"], "metric": "gross_profit",
    }).json()

    assert body["better"] in {"left", "right", "tie"}
    assert body["verdict"]
    assert body["left"]["run_label"] == "均衡"

    assert client.get("/backtest/compare", params={
        "run_a": left["id"], "run_b": 999,
    }).status_code == 404
