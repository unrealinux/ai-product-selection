"""业务编排层：把存储、规则打分与大模型点评串起来。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from . import db, llm
from .config import settings
from .models import Product, ProductIn
from .scoring import grade_of, rule_score


def prepare_db() -> None:
    """确保数据库可用。"""
    db.init_db()


def import_products(products: list[ProductIn]) -> list[Product]:
    """导入候选商品（按标题 + 来源去重）。"""
    return db.bulk_upsert(products)


def score_product(product_id: int, use_llm: bool = True,
                  persist: bool = True) -> Optional[dict[str, Any]]:
    """对单个商品打分。

    Args:
        product_id: 商品主键。
        use_llm: 是否调用大模型点评（配置缺失时自动跳过）。
        persist: 是否写入 scores 表。

    Returns:
        打分明细字典；商品不存在时返回 None。
    """
    product = db.get_product(product_id)
    if product is None:
        return None

    rule = rule_score(product, settings.weights)
    total = rule.total
    review: Optional[str] = None
    adjustment = 0.0

    if use_llm and llm.is_available():
        review, adjustment = llm.review_product(product, rule.dimensions, rule.total)
        total = llm.apply_adjustment(rule.total, adjustment)

    result: dict[str, Any] = {
        "product_id": product.id,
        "title": product.title,
        "category": product.category,
        "total": total,
        "grade": grade_of(total),
        "dimensions": rule.dimensions,
        "profit_margin": rule.profit_margin,
        "advice": rule.advice,
        "llm_review": review,
        "llm_adjustment": adjustment,
        "scored_at": datetime.now(),
        "price": product.price,
        "cost": product.cost,
        "source": product.source,
        "url": product.url,
    }

    if persist:
        db.save_score(result)
    return result


def score_all(use_llm: bool = True, limit: int = 500) -> list[dict[str, Any]]:
    """对库中所有商品打分，返回按总分倒序的结果。"""
    results: list[dict[str, Any]] = []
    for product in db.list_products(limit=limit):
        result = score_product(product.id, use_llm=use_llm)
        if result:
            results.append(result)
    results.sort(key=lambda item: item["total"], reverse=True)
    return results


def leaderboard(limit: int = 50, category: str | None = None) -> list[dict[str, Any]]:
    """选品榜单：优先返回已打分商品，未打分的商品即时打分但不落库。"""
    scored = db.latest_scores(limit=limit * 2)
    if scored:
        if category:
            scored = [item for item in scored if item.get("category") == category]
        return scored[:limit]

    results: list[dict[str, Any]] = []
    for product in db.list_products(category=category, limit=limit):
        result = score_product(product.id, use_llm=False, persist=False)
        if result:
            results.append(result)
    results.sort(key=lambda item: item["total"], reverse=True)
    return results


def dashboard_stats() -> dict[str, Any]:
    """看板统计。"""
    return db.stats()
