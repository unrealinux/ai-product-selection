"""SQLite 存储层：建表、写入商品、保存打分结果、查询排序。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .config import settings
from .models import Product, ProductIn

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    external_id     TEXT    NOT NULL DEFAULT '',
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

-- 权重方案
CREATE TABLE IF NOT EXISTS weight_profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    weights     TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

-- 一次打分运行的快照（权重与商品时刻都固定下来，事后可复现、可对比）
CREATE TABLE IF NOT EXISTS score_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT    NOT NULL,
    profile_id    INTEGER REFERENCES weight_profiles(id) ON DELETE SET NULL,
    weights       TEXT    NOT NULL,
    note          TEXT    NOT NULL DEFAULT '',
    product_count INTEGER NOT NULL DEFAULT 0,
    avg_score     REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS score_run_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES score_runs(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    title      TEXT    NOT NULL DEFAULT '',
    category   TEXT    NOT NULL DEFAULT '',
    source     TEXT    NOT NULL DEFAULT '',
    total      REAL    NOT NULL,
    rank_no    INTEGER NOT NULL,
    dimensions TEXT    NOT NULL,
    UNIQUE (run_id, product_id)
);

CREATE INDEX IF NOT EXISTS idx_run_items_run ON score_run_items(run_id);

-- 表格导入的列映射方案（按源列名集合指纹复用）
CREATE TABLE IF NOT EXISTS mapping_profiles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    fingerprint    TEXT    NOT NULL,
    mapping        TEXT    NOT NULL,
    sample_columns TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mapping_fingerprint ON mapping_profiles(fingerprint);
"""

PRODUCT_FIELDS = (
    "title", "external_id", "category", "price", "cost", "source", "url",
    "heat", "competition", "weight_kg", "repurchase",
    "compliance_risk", "virality", "note",
)

#: 建表后新增的列（列名, ALTER TABLE 片段）。旧库启动时自动补列，避免手动迁移
COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("external_id", "external_id TEXT NOT NULL DEFAULT ''"),
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


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """为已存在的旧库补齐后加的列（幂等）。"""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(products)")}
    for column, ddl in COLUMN_MIGRATIONS:
        if column not in existing:
            conn.execute(f"ALTER TABLE products ADD COLUMN {ddl}")


def init_db(db_path: Path | str | None = None) -> None:
    """初始化表结构（幂等），并补齐旧库缺失的列。"""
    with session(db_path) as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)


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


# --------------------------------------------------------------------------- #
# 权重方案
# --------------------------------------------------------------------------- #

def _profile_row(row: dict[str, Any]) -> dict[str, Any]:
    row["weights"] = json.loads(row.get("weights") or "{}")
    for key in ("created_at", "updated_at"):
        if row.get(key):
            row[key] = datetime.fromisoformat(row[key])
    return row


def upsert_profile(name: str, weights: Mapping[str, float], description: str = "",
                   db_path: Path | str | None = None) -> dict[str, Any]:
    """新增或更新权重方案（按 name 唯一）。"""
    now = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(dict(weights), ensure_ascii=False, sort_keys=True)
    with session(db_path) as conn:
        conn.execute(
            "INSERT INTO weight_profiles (name, weights, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (name) DO UPDATE SET weights = excluded.weights, "
            "description = excluded.description, updated_at = excluded.updated_at",
            (name, payload, description, now, now),
        )
        row = conn.execute(
            "SELECT * FROM weight_profiles WHERE name = ?", (name,)
        ).fetchone()
    return _profile_row(dict(row))


def list_profiles(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    with session(db_path) as conn:
        rows = conn.execute("SELECT * FROM weight_profiles ORDER BY id").fetchall()
    return [_profile_row(dict(row)) for row in rows]


def get_profile(name_or_id: str | int,
                db_path: Path | str | None = None) -> Optional[dict[str, Any]]:
    """按名字或 id 取方案。

    **先按名字查**：方案名很可能是纯数字（例如「1688」），
    直接当 id 处理会查不到——这类静默失败很难排查。
    """
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM weight_profiles WHERE name = ?", (str(name_or_id),)
        ).fetchone()
        if row is None and (isinstance(name_or_id, int) or str(name_or_id).isdigit()):
            row = conn.execute(
                "SELECT * FROM weight_profiles WHERE id = ?", (int(name_or_id),)
            ).fetchone()
    return _profile_row(dict(row)) if row else None


def delete_profile(name_or_id: str | int, db_path: Path | str | None = None) -> bool:
    profile = get_profile(name_or_id, db_path)
    if not profile:
        return False
    with session(db_path) as conn:
        conn.execute("DELETE FROM weight_profiles WHERE id = ?", (profile["id"],))
    return True


# --------------------------------------------------------------------------- #
# 打分运行快照
# --------------------------------------------------------------------------- #

def _run_row(row: dict[str, Any]) -> dict[str, Any]:
    row["weights"] = json.loads(row.get("weights") or "{}")
    if row.get("created_at"):
        row["created_at"] = datetime.fromisoformat(row["created_at"])
    return row


def create_run(label: str, weights: Mapping[str, float], items: list[dict[str, Any]],
               profile_id: Optional[int] = None, note: str = "",
               db_path: Path | str | None = None) -> dict[str, Any]:
    """把一次打分结果固化成快照。

    Args:
        items: 每项至少包含 ``product_id`` 与 ``total``；可选 ``title`` / ``category`` /
            ``source`` / ``dimensions``。排名按 ``total`` 倒序自动生成。

    快照同时存下当时的权重与各维度得分，因此后续商品数据或方案被修改都不影响历史对比。
    """
    ordered = sorted(items, key=lambda item: item["total"], reverse=True)
    average = round(sum(item["total"] for item in ordered) / len(ordered), 2) if ordered else 0.0
    now = datetime.now().isoformat(timespec="seconds")

    with session(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO score_runs (label, profile_id, weights, note, product_count, "
            "avg_score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (label, profile_id,
             json.dumps(dict(weights), ensure_ascii=False, sort_keys=True),
             note, len(ordered), average, now),
        )
        run_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO score_run_items (run_id, product_id, title, category, source, "
            "total, rank_no, dimensions) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (run_id, item["product_id"], item.get("title", ""),
                 item.get("category", ""), item.get("source", ""),
                 float(item["total"]), index,
                 json.dumps(item.get("dimensions") or {}, ensure_ascii=False))
                for index, item in enumerate(ordered, start=1)
            ],
        )
        row = conn.execute("SELECT * FROM score_runs WHERE id = ?", (run_id,)).fetchone()
    return _run_row(dict(row))


def list_runs(limit: int = 50, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    with session(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM score_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_run_row(dict(row)) for row in rows]


def get_run(run_id: int, db_path: Path | str | None = None) -> Optional[dict[str, Any]]:
    with session(db_path) as conn:
        row = conn.execute("SELECT * FROM score_runs WHERE id = ?", (run_id,)).fetchone()
    return _run_row(dict(row)) if row else None


def get_run_items(run_id: int, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """取快照内的条目，按排名升序。"""
    with session(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM score_run_items WHERE run_id = ? ORDER BY rank_no", (run_id,)
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["dimensions"] = json.loads(item.get("dimensions") or "{}")
        items.append(item)
    return items


def delete_run(run_id: int, db_path: Path | str | None = None) -> bool:
    with session(db_path) as conn:
        cursor = conn.execute("DELETE FROM score_runs WHERE id = ?", (run_id,))
    return cursor.rowcount > 0


# --------------------------------------------------------------------------- #
# 表格导入的列映射方案
# --------------------------------------------------------------------------- #

def _mapping_row(row: dict[str, Any]) -> dict[str, Any]:
    row["mapping"] = json.loads(row.get("mapping") or "{}")
    row["columns"] = [c for c in (row.get("sample_columns") or "").split("\x1f") if c]
    for key in ("created_at", "updated_at"):
        if row.get(key):
            row[key] = datetime.fromisoformat(row[key])
    return row


def upsert_mapping_profile(name: str, fingerprint: str, mapping: Mapping[str, Any],
                           columns: Optional[list[str]] = None,
                           db_path: Path | str | None = None) -> dict[str, Any]:
    """保存列映射方案（按 name 唯一）。"""
    now = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(dict(mapping), ensure_ascii=False, sort_keys=True)
    sample = "\x1f".join(columns or [])
    with session(db_path) as conn:
        conn.execute(
            "INSERT INTO mapping_profiles (name, fingerprint, mapping, sample_columns, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (name) DO UPDATE SET fingerprint = excluded.fingerprint, "
            "mapping = excluded.mapping, sample_columns = excluded.sample_columns, "
            "updated_at = excluded.updated_at",
            (name, fingerprint, payload, sample, now, now),
        )
        row = conn.execute(
            "SELECT * FROM mapping_profiles WHERE name = ?", (name,)
        ).fetchone()
    return _mapping_row(dict(row))


def list_mapping_profiles(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    with session(db_path) as conn:
        rows = conn.execute("SELECT * FROM mapping_profiles ORDER BY id").fetchall()
    return [_mapping_row(dict(row)) for row in rows]


def get_mapping_profile(name_or_id: str | int,
                        db_path: Path | str | None = None) -> Optional[dict[str, Any]]:
    """按名字或 id 取映射方案。**先按名字查**（名字可能是「1688」这种纯数字）。"""
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM mapping_profiles WHERE name = ?", (str(name_or_id),)
        ).fetchone()
        if row is None and (isinstance(name_or_id, int) or str(name_or_id).isdigit()):
            row = conn.execute(
                "SELECT * FROM mapping_profiles WHERE id = ?", (int(name_or_id),)
            ).fetchone()
    return _mapping_row(dict(row)) if row else None


def find_mapping_by_fingerprint(fingerprint: str,
                                db_path: Path | str | None = None) -> Optional[dict[str, Any]]:
    """按源列名指纹找最合适的已存方案（列集合一致时优先，其次取最新）。"""
    with session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM mapping_profiles WHERE fingerprint = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT 1",
            (fingerprint,),
        ).fetchone()
    return _mapping_row(dict(row)) if row else None


def delete_mapping_profile(name_or_id: str | int,
                           db_path: Path | str | None = None) -> bool:
    record = get_mapping_profile(name_or_id, db_path)
    if not record:
        return False
    with session(db_path) as conn:
        conn.execute("DELETE FROM mapping_profiles WHERE id = ?", (record["id"],))
    return True
