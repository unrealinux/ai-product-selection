"""业务编排层：把存储、规则打分与大模型点评串起来。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from . import db, llm, outcomes
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


# --------------------------------------------------------------------------- #
# 表格导入（CSV / Excel）
# --------------------------------------------------------------------------- #

def preview_table(path, mapping: Optional[dict[str, Any]] = None,
                  db_path: Any = None, **options: Any) -> tuple[Any, Any, Any]:
    """读表 + 确定列映射 + 试算，不写库。供 UI 预览与确认。

    映射优先级：显式传入 > 指纹命中的已存方案 > 自动推断。

    Returns:
        ``(TableData, ColumnMapping, BuildResult)``
    """
    from . import tabular

    source = tabular.TabularSource(path, mapping=mapping, **options)
    data = source.load()

    resolved = None
    if mapping is None:
        record = db.find_mapping_by_fingerprint(
            tabular.column_fingerprint(data.columns), db_path
        )
        if record:
            resolved = tabular.ColumnMapping.from_dict(record["mapping"])
            resolved.source = source.source_label or resolved.source
    if resolved is None:
        resolved = source.resolve_mapping(data.columns)

    return data, resolved, tabular.build_products(data, resolved)


def import_table(path, mapping: Optional[dict[str, Any]] = None,
                 save_as: str = "", db_path: Any = None,
                 **options: Any) -> dict[str, Any]:
    """把表格文件导入商品库。

    Args:
        mapping: 显式列映射；为空时先查指纹缓存的方案，再退到自动推断。
        save_as: 非空时把本次使用的映射存为该名字的方案，下次自动复用。
    """
    from . import tabular

    data, resolved, result = preview_table(path, mapping=mapping,
                                           db_path=db_path, **options)
    if not result.products:
        detail = "；".join(result.warnings) if result.warnings else "未解析出任何商品"
        raise ValueError(detail)

    if save_as:
        db.upsert_mapping_profile(
            save_as, tabular.column_fingerprint(data.columns),
            resolved.to_dict(), data.columns, db_path,
        )

    products = tabular.apply_provenance(result.products, result.provenance)
    saved = db.bulk_upsert(products, db_path)
    return {
        "table": data.describe(),
        "columns": list(data.columns),
        "mapping": resolved.to_dict(),
        "saved": len(saved),
        "skipped": result.skipped,
        "warnings": list(result.warnings),
        "provenance": result.provenance,
    }


# --------------------------------------------------------------------------- #
# 效果回测：经营结果录入与「哪套权重更赚钱」
# --------------------------------------------------------------------------- #


def _parse_day(value: Any, name: str) -> str:
    """把日期统一成 ``YYYY-MM-DD``；解析不了就报错，不静默落库。"""
    parsed = outcomes.parse_day(value)
    if parsed is None:
        raise ValueError(f"{name} 需要 YYYY-MM-DD 格式，收到 {value!r}")
    return parsed


def record_outcome(product_id: int, window_start: Any, window_end: Any,
                   **fields: Any) -> dict[str, Any]:
    """录入一段经营结果（商品必须存在，数值不得为负）。"""
    if db.get_product(product_id) is None:
        raise KeyError(f"商品 {product_id} 不存在")

    start = _parse_day(window_start, "window_start")
    end = _parse_day(window_end, "window_end")
    if start > end:
        raise ValueError("window_start 不能晚于 window_end")

    negative = [
        name for name in db.OUTCOME_NUMERIC_FIELDS
        if name in fields and float(fields.get(name) or 0) < 0
    ]
    if negative:
        raise ValueError("结果数值不能为负：" + "、".join(negative))

    return db.upsert_outcome(product_id, start, end, **fields)


def list_outcomes(product_id: Optional[int] = None,
                  limit: int = 1000) -> list[dict[str, Any]]:
    return db.list_outcomes(product_id=product_id, limit=limit)


def delete_outcome(outcome_id: int) -> bool:
    return db.delete_outcome(outcome_id)


def record_decision(product_id: int, run_id: Optional[int] = None,
                    action: str = "push", note: str = "") -> dict[str, Any]:
    """记录一次选品决策（推 / 压 / 拒）。"""
    if db.get_product(product_id) is None:
        raise KeyError(f"商品 {product_id} 不存在")
    if action not in {"push", "hold", "skip"}:
        raise ValueError(f"未知行动 {action!r}，可选：push / hold / skip")
    if run_id is not None and db.get_run(run_id) is None:
        raise KeyError(f"快照 {run_id} 不存在")
    return db.record_decision(product_id, run_id=run_id, action=action, note=note)


def list_decisions(run_id: Optional[int] = None,
                   product_id: Optional[int] = None) -> list[dict[str, Any]]:
    return db.list_decisions(run_id=run_id, product_id=product_id)


def backtest_run(run_id: int, metric: str = outcomes.DEFAULT_METRIC,
                 top_ratio: float = 0.3,
                 after_run_only: bool = False) -> Any:
    """把一次打分快照与已录入的经营结果对照。

    ``after_run_only=True`` 时只采用快照创建之后的结果窗口 ——
    打分发生在结果之后就不构成预测。
    """
    run = db.get_run(run_id)
    if run is None:
        raise KeyError(f"快照 {run_id} 不存在")
    if metric not in outcomes.METRICS:
        # 指标名写错是调用方的问题，应与「快照不存在」区分开
        raise ValueError(f"未知指标 {metric!r}，可选：{', '.join(outcomes.METRICS)}")
    return outcomes.backtest(
        run, db.get_run_items(run_id), db.list_outcomes(),
        metric=metric, top_ratio=top_ratio, after_run_only=after_run_only,
    )


def compare_backtests(run_a: int, run_b: int, metric: str = outcomes.DEFAULT_METRIC,
                      top_ratio: float = 0.3,
                      after_run_only: bool = False) -> Any:
    """对比两次快照谁更能预测同一项经营指标。"""
    left = backtest_run(run_a, metric=metric, top_ratio=top_ratio,
                        after_run_only=after_run_only)
    right = backtest_run(run_b, metric=metric, top_ratio=top_ratio,
                         after_run_only=after_run_only)
    return outcomes.compare_backtests(left, right)
