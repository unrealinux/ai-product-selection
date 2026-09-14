"""FastAPI 服务：对外暴露选品库的导入、打分与榜单接口。

启动：
    uvicorn app.api:app --reload
或：
    python -m app.api
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__, crawler, db, llm, service
from .config import settings
from .models import Product, ProductIn
from .sources.douyin import ERROR_CODES, DouyinClient, DouyinError

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
