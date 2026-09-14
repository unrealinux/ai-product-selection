"""SQLite 存储层：建表、写入商品、保存打分结果、查询排序。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import settings
from .models import Product, ProductIn

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    category        TEXT    NOT NULL DEFAULT '未分类',
    price           REAL    NOT NULL DEFAULT 0,
    cost            REAL    NOT NULL DEFAULT 0,
    source          TEXT    NOT NULL DEFAULT 'manual',
    url             TEXT    NOT NULL DEFAULT '',
    heat            REAL    NOT NULL DEFAULT 50,
    competition     REAL    NOT NULL DEFAULT 50,
    weight_kg       REAL    NOT NULL DEFAULT 0.5,
    repurchase      REAL    NOT NULL DEFAULT 50,
    compliance_risk REAL    NOT NULL DEFAULT 20,
    virality        REAL    NOT NULL DEFAULT 50,
    note            TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL,
    UNIQUE (title, source)
);

CREATE TABLE IF NOT EXISTS scores (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id     INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    total          REAL    NOT NULL,
    grade          TEXT    NOT NULL,
    dimensions     TEXT    NOT NULL,
    profit_margin  REAL    NOT NULL DEFAULT 0,
    advice         TEXT    NOT NULL DEFAULT '',
    llm_review     TEXT,
    llm_adjustment REAL    NOT NULL DEFAULT 0,
    scored_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_product ON scores(product_id);
CREATE INDEX IF NOT EXISTS idx_scores_total   ON scores(total DESC);
"""

PRODUCT_FIELDS = (
    "title", "category", "price", "cost", "source", "url",
    "heat", "competition", "weight_kg", "repurchase",
    "compliance_risk", "virality", "note",
)


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """打开数据库连接（自动创建目录、开启外键与 WAL）。"""
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def session(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """上下文管理器，自动提交 / 回滚并关闭连接。"""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """初始化表结构（幂等）。"""
    with session(db_path) as conn:
        conn.executescript(SCHEMA)


def _row_to_product(row: sqlite3.Row) -> Product:
    data = dict(row)
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    return Product(**data)


def upsert_product(product: ProductIn, db_path: Path | str | None = None) -> Product:
    """按 (title, source) 去重写入商品，返回最终记录。"""
    now = datetime.now().isoformat(timespec="seconds")
    values = [getattr(product, field) for field in PRODUCT_FIELDS]
    placeholders = ", ".join("?" for _ in PRODUCT_FIELDS)
    updates = ", ".join(f"{field}=excluded.{field}" for field in PRODUCT_FIELDS if field != "title")

    with session(db_path) as conn:
        conn.execute(
            f"INSERT INTO products ({', '.join(PRODUCT_FIELDS)}, created_at) "
            f"VALUES ({placeholders}, ?) "
            f"ON CONFLICT (title, source) DO UPDATE SET {updates}",
            [*values, now],
        )
        row = conn.execute(
            "SELECT * FROM products WHERE title = ? AND source = ?",
            (product.title, product.source),
        ).fetchone()
    return _row_to_product(row)


def bulk_upsert(products: list[ProductIn], db_path: Path | str | None = None) -> list[Product]:
    """批量写入商品。"""
    return [upsert_product(item, db_path) for item in products]


def get_product(product_id: int, db_path: Path | str | None = None) -> Optional[Product]:
    with session(db_path) as conn:
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    return _row_to_product(row) if row else None


def list_products(category: str | None = None, source: str | None = None,
                  limit: int = 200, db_path: Path | str | None = None) -> list[Product]:
    clauses, params = [], []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if source:
        clauses.append("source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with session(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM products {where} ORDER BY id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
    return [_row_to_product(row) for row in rows]


def save_score(result: dict[str, Any], db_path: Path | str | None = None) -> None:
    """保存一次打分结果（保留历史，查询时取最新一条）。"""
    with session(db_path) as conn:
        conn.execute(
            "INSERT INTO scores (product_id, total, grade, dimensions, profit_margin, "
            "advice, llm_review, llm_adjustment, scored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result["product_id"],
                result["total"],
                result["grade"],
                json.dumps(result.get("dimensions", {}), ensure_ascii=False),
                result.get("profit_margin", 0.0),
                result.get("advice", ""),
                result.get("llm_review"),
                result.get("llm_adjustment", 0.0),
                result.get("scored_at", datetime.now()).isoformat(timespec="seconds")
                if isinstance(result.get("scored_at"), datetime)
                else str(result.get("scored_at") or datetime.now().isoformat(timespec="seconds")),
            ),
        )


def latest_scores(limit: int = 500, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """取每个商品的最新一次打分，按总分倒序。"""
    with session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*, p.title, p.category, p.price, p.cost, p.source, p.url
            FROM scores s
            JOIN products p ON p.id = s.product_id
            WHERE s.id = (SELECT MAX(id) FROM scores WHERE product_id = s.product_id)
            ORDER BY s.total DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_score_row(dict(row)) for row in rows]


def latest_score(product_id: int, db_path: Path | str | None = None) -> Optional[dict[str, Any]]:
    """取单个商品的最新打分。"""
    with session(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.*, p.title, p.category, p.price, p.cost, p.source, p.url
            FROM scores s
            JOIN products p ON p.id = s.product_id
            WHERE s.product_id = ?
            ORDER BY s.id DESC LIMIT 1
            """,
            (product_id,),
        ).fetchone()
    return _score_row(dict(row)) if row else None


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    row["dimensions"] = json.loads(row.get("dimensions") or "{}")
    row["scored_at"] = datetime.fromisoformat(row["scored_at"])
    return row


def stats(db_path: Path | str | None = None) -> dict[str, Any]:
    """看板统计数据。"""
    with session(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        scored = conn.execute(
            "SELECT COUNT(DISTINCT product_id) FROM scores"
        ).fetchone()[0]
        avg_row = conn.execute(
            """
            SELECT AVG(total) FROM scores
            WHERE id IN (SELECT MAX(id) FROM scores GROUP BY product_id)
            """
        ).fetchone()[0]
        grade_rows = conn.execute(
            """
            SELECT grade, COUNT(*) AS n FROM scores
            WHERE id IN (SELECT MAX(id) FROM scores GROUP BY product_id)
            GROUP BY grade
            """
        ).fetchall()
        category_rows = conn.execute(
            """
            SELECT p.category, COUNT(*) AS n, ROUND(AVG(s.total), 1) AS avg_score
            FROM scores s
            JOIN products p ON p.id = s.product_id
            WHERE s.id = (SELECT MAX(id) FROM scores WHERE product_id = s.product_id)
            GROUP BY p.category
            ORDER BY avg_score DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "total": total,
        "scored": scored,
        "avg_score": round(avg_row or 0.0, 2),
        "grade_distribution": {row["grade"]: row["n"] for row in grade_rows},
        "top_categories": [dict(row) for row in category_rows],
    }
