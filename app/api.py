"""FastAPI 服务：对外暴露选品库的导入、打分与榜单接口。

启动：
    uvicorn app.api:app --reload
或：
    python -m app.api
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__, crawler, db, llm, outcomes, service
from .config import settings
from .models import Product, ProductIn
from .sources.douyin import ERROR_CODES, DouyinClient, DouyinError
from .weights import PRESETS

app = FastAPI(
    title="AI 选品库",
    description="规则引擎 + 大模型的多维度选品打分服务",
    version=__version__,
)


class ImportRequest(BaseModel):
    """导入请求。"""

    source: str = Field("sample", description="数据源：sample / json / douyin")
    path: Optional[str] = Field(None, description="json 源的文件路径；douyin 源可传逗号分隔关键词")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description='数据源构造参数，如 {"page_size": 20, "max_pages": 2}',
    )
    score: bool = Field(True, description="导入后是否立即打分")
    use_llm: bool = Field(True, description="是否启用大模型点评")


class ImportResponse(BaseModel):
    imported: int
    scored: int
    items: list[dict[str, Any]]


@app.on_event("startup")
def _startup() -> None:
    service.prepare_db()


@app.get("/health", summary="健康检查")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "llm_available": llm.is_available(),
        "db": str(settings.db_path),
    }


@app.get("/products", response_model=list[Product], summary="商品列表")
def list_products(
    category: Optional[str] = Query(None, description="按类目过滤"),
    source: Optional[str] = Query(None, description="按来源过滤"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[Product]:
    return db.list_products(category=category, source=source, limit=limit)


@app.get("/products/{product_id}", response_model=Product, summary="商品详情")
def get_product(product_id: int) -> Product:
    product = db.get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"商品 {product_id} 不存在")
    return product


@app.post("/products", response_model=Product, status_code=201, summary="新增商品")
def create_product(product: ProductIn) -> Product:
    service.prepare_db()
    return db.upsert_product(product)


@app.post("/import", response_model=ImportResponse, summary="从数据源批量导入")
def import_products(payload: ImportRequest) -> ImportResponse:
    service.prepare_db()
    try:
        source = crawler.get_source(payload.source, payload.path, payload.options)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DouyinError as exc:
        # 凭据缺失 / 签名失败 / 限流等，都属于调用方可修复的问题
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        products = source.fetch()
    except FileNotFoundError as exc:
        # 数据文件在 fetch 阶段才被打开，这里也必须映射成 404
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DouyinError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    saved = service.import_products(products)

    items: list[dict[str, Any]] = []
    if payload.score:
        for product in saved:
            result = service.score_product(product.id, use_llm=payload.use_llm)
            if result:
                items.append(result)
        items.sort(key=lambda item: item["total"], reverse=True)

    return ImportResponse(imported=len(saved), scored=len(items), items=items)


@app.post("/score/{product_id}", summary="给单个商品打分")
def score_one(product_id: int, use_llm: bool = Query(True)) -> dict[str, Any]:
    result = service.score_product(product_id, use_llm=use_llm)
    if result is None:
        raise HTTPException(status_code=404, detail=f"商品 {product_id} 不存在")
    return result


@app.post("/score", summary="全库打分")
def score_all(use_llm: bool = Query(True), limit: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
    results = service.score_all(use_llm=use_llm, limit=limit)
    return {"scored": len(results), "items": results}


@app.get("/leaderboard", summary="选品榜单")
def leaderboard(
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    return service.leaderboard(limit=limit, category=category)


@app.get("/stats", summary="看板统计")
def stats() -> dict[str, Any]:
    return service.dashboard_stats()


# --------------------------------------------------------------------------- #
# 权重方案与打分快照（调参 / A/B 对比）
# --------------------------------------------------------------------------- #

class ProfileRequest(BaseModel):
    """保存权重方案。"""

    name: str = Field(..., min_length=1, max_length=64)
    weights: dict[str, float] = Field(..., description="维度名 → 权重，会自动归一化")
    description: str = Field("", max_length=200)


class RunRequest(BaseModel):
    """按指定权重打一次分并固化成快照。"""

    label: str = Field(..., min_length=1, max_length=100, description="快照名称")
    weights: Optional[dict[str, float]] = Field(
        None, description="直接指定权重；与 profile 二选一"
    )
    profile: Optional[str] = Field(
        None, description="权重方案名或内置预设名（balanced / margin_first / ...）"
    )
    note: str = Field("", max_length=200, description="备注，例如「接口热度」")


def _run_brief(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": run["id"],
        "label": run["label"],
        "profile_id": run.get("profile_id"),
        "weights": run["weights"],
        "note": run.get("note", ""),
        "product_count": run["product_count"],
        "avg_score": run["avg_score"],
        "created_at": run.get("created_at"),
    }


def _comparison_payload(result: Any) -> dict[str, Any]:
    return {
        "run_a": _run_brief(result.run_a),
        "run_b": _run_brief(result.run_b),
        "common": result.common,
        "only_a": result.only_a,
        "only_b": result.only_b,
        "spearman": result.spearman,
        "verdict": result.verdict,
        "summary": result.summary(),
        "avg_abs_rank_delta": result.avg_abs_rank_delta,
        "max_rank_delta": result.max_rank_delta,
        "top_overlap": result.top_overlap,
        "weight_diff": result.weight_diff,
        "notes": result.notes,
        "movers": [
            {**asdict(mover), "rank_delta": mover.rank_delta,
             "score_delta": mover.score_delta, "direction": mover.direction}
            for mover in result.movers
        ],
    }


@app.get("/presets", summary="内置权重预设")
def list_presets() -> list[dict[str, Any]]:
    return [
        {"name": name, "label": spec["label"],
         "description": spec["description"], "weights": spec["weights"]}
        for name, spec in PRESETS.items()
    ]


@app.get("/profiles", summary="权重方案列表")
def list_profiles() -> list[dict[str, Any]]:
    return db.list_profiles()


@app.post("/profiles", status_code=201, summary="保存权重方案")
def create_profile(payload: ProfileRequest) -> dict[str, Any]:
    try:
        return service.save_profile(payload.name, payload.weights, payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/profiles/install-presets", summary="把内置预设写入数据库")
def install_presets() -> dict[str, Any]:
    return {"installed": service.install_presets()}


@app.delete("/profiles/{name}", summary="删除权重方案")
def remove_profile(name: str) -> dict[str, Any]:
    if not db.delete_profile(name):
        raise HTTPException(status_code=404, detail=f"方案 {name!r} 不存在")
    return {"deleted": name}


@app.get("/runs", summary="打分快照列表")
def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    return [_run_brief(run) for run in service.list_snapshots(limit=limit)]


@app.post("/runs", status_code=201, summary="打分并固化快照")
def create_run(payload: RunRequest) -> dict[str, Any]:
    service.prepare_db()
    try:
        run = service.create_snapshot(
            payload.label, weights=payload.weights,
            profile=payload.profile, note=payload.note,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _run_brief(run)


@app.get("/runs/{run_id}", summary="快照详情（含排名）")
def get_run(run_id: int) -> dict[str, Any]:
    try:
        run = service.snapshot_detail(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {**_run_brief(run), "items": run["items"]}


@app.delete("/runs/{run_id}", summary="删除快照")
def remove_run(run_id: int) -> dict[str, Any]:
    if not db.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"快照 {run_id} 不存在")
    return {"deleted": run_id}


@app.get("/runs/{run_id}/sensitivity", summary="该快照下各维度的影响力")
def run_sensitivity(run_id: int, top_n: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    try:
        impacts = service.snapshot_sensitivity(run_id, top_n=top_n)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    note = ""
    if not impacts:
        note = (
            "该快照只有一个非零维度，把它归零后所有商品分数相同、排名无意义，"
            "因此没有可计算的影响力数据。请先用包含多个维度的权重创建快照。"
        )
    return {
        "run_id": run_id,
        "top_n": top_n,
        "note": note,
        "impacts": [
            {**asdict(impact), "influence": impact.influence, "verdict": impact.verdict}
            for impact in impacts
        ],
    }


@app.get("/compare", summary="对比两个快照（A/B）")
def compare(
    run_a: int = Query(..., description="快照 A 的 id"),
    run_b: int = Query(..., description="快照 B 的 id"),
    movers: int = Query(10, ge=1, le=200, description="返回多少个变动最大的商品"),
) -> dict[str, Any]:
    try:
        result = service.compare_snapshots(run_a, run_b, movers=movers)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _comparison_payload(result)


# --------------------------------------------------------------------------- #
# 效果回测：经营结果 + 决策记录
# --------------------------------------------------------------------------- #

class OutcomeRequest(BaseModel):
    """录入一段经营结果。

    同一 ``(product_id, window_start, window_end)`` 重复提交会更新而非新增。
    """

    product_id: int = Field(..., description="商品主键")
    window_start: date = Field(..., description="统计开始日 YYYY-MM-DD")
    window_end: date = Field(..., description="统计结束日 YYYY-MM-DD")
    impressions: float = Field(0.0, ge=0, description="曝光")
    clicks: float = Field(0.0, ge=0, description="点击")
    orders: float = Field(0.0, ge=0, description="订单数")
    units: float = Field(0.0, ge=0, description="销量（件）")
    returns: float = Field(0.0, ge=0, description="退货件数")
    revenue: float = Field(0.0, ge=0, description="成交金额（元）")
    cogs: float = Field(0.0, ge=0, description="实际采购成本（元）")
    ad_spend: float = Field(0.0, ge=0, description="推广花费（元）")
    note: str = Field("", max_length=500)
    source: str = Field("manual", max_length=32, description="数据来源，如 生意参谋导出")


class DecisionRequest(BaseModel):
    """记录一次选品决策。"""

    product_id: int
    run_id: Optional[int] = Field(None, description="关联的打分快照 id")
    action: str = Field("push", pattern="^(push|hold|skip)$", description="push / hold / skip")
    note: str = Field("", max_length=500)


@app.get("/metrics", summary="回测可用指标")
def list_metrics() -> list[dict[str, Any]]:
    return [
        {"name": name, "label": spec["label"], "higher_better": spec["higher_better"]}
        for name, spec in outcomes.METRICS.items()
    ]


@app.post("/outcomes", status_code=201, summary="录入经营结果")
def create_outcome(payload: OutcomeRequest) -> dict[str, Any]:
    try:
        return service.record_outcome(**payload.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/outcomes", summary="经营结果列表")
def list_outcomes(
    product_id: Optional[int] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    return service.list_outcomes(product_id=product_id, limit=limit)


@app.delete("/outcomes/{outcome_id}", summary="删除经营结果")
def remove_outcome(outcome_id: int) -> dict[str, Any]:
    if not service.delete_outcome(outcome_id):
        raise HTTPException(status_code=404, detail=f"结果记录 {outcome_id} 不存在")
    return {"deleted": outcome_id}


@app.post("/decisions", status_code=201, summary="记录选品决策")
def create_decision(payload: DecisionRequest) -> dict[str, Any]:
    try:
        return service.record_decision(**payload.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/decisions", summary="决策记录列表")
def list_decisions(
    run_id: Optional[int] = Query(None),
    product_id: Optional[int] = Query(None),
) -> list[dict[str, Any]]:
    return service.list_decisions(run_id=run_id, product_id=product_id)


@app.get("/runs/{run_id}/backtest", summary="用真实经营结果回测该快照")
def run_backtest(
    run_id: int,
    metric: str = Query(outcomes.DEFAULT_METRIC, description="指标名，见 /metrics"),
    top_ratio: float = Query(0.3, gt=0, le=1, description="Top 组占比"),
    after_run_only: bool = Query(
        False, description="只采用快照创建之后的结果窗口（推荐：打分之后的结果才构成预测）"
    ),
) -> dict[str, Any]:
    try:
        result = service.backtest_run(run_id, metric=metric, top_ratio=top_ratio,
                                      after_run_only=after_run_only)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.as_dict()


@app.get("/backtest/compare", summary="对比两次快照的预测力")
def compare_backtest(
    run_a: int = Query(..., description="快照 A 的 id"),
    run_b: int = Query(..., description="快照 B 的 id"),
    metric: str = Query(outcomes.DEFAULT_METRIC),
    top_ratio: float = Query(0.3, gt=0, le=1),
    after_run_only: bool = Query(False),
) -> dict[str, Any]:
    try:
        comparison = service.compare_backtests(
            run_a, run_b, metric=metric, top_ratio=top_ratio,
            after_run_only=after_run_only,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return comparison.as_dict()


# --------------------------------------------------------------------------- #
# 抖音（抖店）数据源
# --------------------------------------------------------------------------- #

class DouyinTokenRequest(BaseModel):
    """换取 access_token 的请求。

    自用型应用只需 ``shop_id``；店铺授权码模式传 ``code``。
    """

    shop_id: Optional[str] = Field(None, description="自用型应用必填")
    code: Optional[str] = Field(None, description="授权码模式必填")


@app.get("/douyin/status", summary="抖音数据源配置状态")
def douyin_status() -> dict[str, Any]:
    return {
        "configured": settings.douyin_ready,
        "has_app_key": bool(settings.douyin_app_key),
        "has_app_secret": bool(settings.douyin_app_secret),
        "has_access_token": bool(settings.douyin_access_token),
        "shop_id": settings.douyin_shop_id or None,
        "sign_method": settings.douyin_sign_method,
        "base_url": settings.douyin_base_url,
        "error_codes": ERROR_CODES,
    }


@app.post("/douyin/token", summary="换取 access_token（结果不落盘）")
def douyin_token(payload: DouyinTokenRequest) -> dict[str, Any]:
    """仅供调试；生产环境请把 token 写入配置，不要每次请求都重新换取。"""
    try:
        client = DouyinClient.from_settings()
        data = (
            client.create_token_by_code(payload.code)
            if payload.code
            else client.create_self_token(payload.shop_id)
        )
    except DouyinError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = str(data.get("access_token") or "")
    return {
        "access_token_preview": token[:8] + "..." if token else None,
        "expires_in": data.get("expires_in"),
        "refresh_token_present": bool(data.get("refresh_token")),
        "hint": "请把 access_token 写入 .env 的 APS_DOUYIN_ACCESS_TOKEN",
    }


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("app.api:app", host=settings.api_host, port=settings.api_port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
