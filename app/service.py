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


# --------------------------------------------------------------------------- #
# 权重方案与打分快照（用于权重调参与 A/B 对比）
# --------------------------------------------------------------------------- #

def _resolve_weights(
    weights: Optional[dict[str, float]] = None,
    profile: Optional[str] = None,
) -> tuple[dict[str, float], Optional[int]]:
    """确定本次使用的权重，返回 ``(权重, profile_id)``。

    优先级：显式 weights > 库里的方案 > 内置预设 > 全局默认。
    """
    from . import weights as weights_mod

    if weights:
        problems = weights_mod.validate(weights)
        if problems:
            raise ValueError("；".join(problems))
        return weights_mod.normalize(weights), None

    if profile:
        record = db.get_profile(profile)
        if record:
            return weights_mod.normalize(record["weights"]), record["id"]
        if profile in weights_mod.PRESETS:
            return weights_mod.preset_weights(profile), None
        raise KeyError(f"未找到权重方案 {profile!r}")

    return dict(settings.weights), None


def save_profile(name: str, weights: dict[str, float], description: str = "") -> dict[str, Any]:
    """保存权重方案（校验失败抛 ValueError）。"""
    from . import weights as weights_mod

    problems = weights_mod.validate(weights)
    if problems:
        raise ValueError("；".join(problems))
    return db.upsert_profile(name, weights_mod.normalize(weights), description)


def install_presets() -> int:
    """把内置预设写入数据库（幂等），返回写入数量。"""
    from . import weights as weights_mod

    for name, spec in weights_mod.PRESETS.items():
        db.upsert_profile(name, weights_mod.normalize(spec["weights"]), spec["description"])
    return len(weights_mod.PRESETS)


def create_snapshot(
    label: str,
    weights: Optional[dict[str, float]] = None,
    profile: Optional[str] = None,
    note: str = "",
    limit: int = 2000,
) -> dict[str, Any]:
    """按指定权重对全库打分，并把结果固化成快照。

    快照会同时存下当时的权重与各维度得分，因此后续商品数据或权重方案变动
    都不会影响历史对比 —— 这是做「接口热度 vs 大模型热度」这类实验的前提。
    """
    resolved, profile_id = _resolve_weights(weights, profile)
    items: list[dict[str, Any]] = []
    for product in db.list_products(limit=limit):
        rule = rule_score(product, resolved)
        items.append({
            "product_id": product.id,
            "title": product.title,
            "category": product.category,
            "source": product.source,
            "url": product.url,
            "total": rule.total,
            "dimensions": rule.dimensions,
        })
    if not items:
        raise ValueError("库里没有商品，无法创建快照。请先导入数据。")
    return db.create_run(label, resolved, items, profile_id=profile_id, note=note)


def list_snapshots(limit: int = 50) -> list[dict[str, Any]]:
    return db.list_runs(limit=limit)


def snapshot_detail(run_id: int) -> dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise KeyError(f"快照 {run_id} 不存在")
    run["items"] = db.get_run_items(run_id)
    return run


def compare_snapshots(run_a: int, run_b: int, movers: int = 10) -> Any:
    """对比两个快照。"""
    from .compare import compare_runs

    left, right = db.get_run(run_a), db.get_run(run_b)
    if left is None:
        raise KeyError(f"快照 {run_a} 不存在")
    if right is None:
        raise KeyError(f"快照 {run_b} 不存在")
    return compare_runs(
        left, db.get_run_items(run_a), right, db.get_run_items(run_b), movers=movers
    )


def snapshot_sensitivity(run_id: int, top_n: int = 10) -> list[Any]:
    """某个快照下各维度对排序的影响力。"""
    from .compare import weight_sensitivity

    run = db.get_run(run_id)
    if run is None:
        raise KeyError(f"快照 {run_id} 不存在")
    return weight_sensitivity(db.get_run_items(run_id), run["weights"], top_n=top_n)
