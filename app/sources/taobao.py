"""淘宝 A2A 数据源。

走的是淘宝桌面客户端内置的 **官方 A2A（Agent2Agent）服务端**，不是页面抓取、也不是
本地那个内测门槛拦着的 ``taobao-native`` CLI。

契约来源（公开可访问，已实测）：

* agent card —— ``{base}/.well-known/agent.json``
* 接口        —— ``POST {base}``，JSON-RPC ``tasks/send``

实测探明的三个 skill：

===================================  ==========================================
入参                                  返回 artifact
===================================  ==========================================
``{"query":..., "sort":"sales_desc"}``  ``search-candidates``
``{"skillId":"item-detail", "itemIds":[...]}``  ``item-detail-result``
``{"skillId":"item-compare", "itemIds":[...]}``  对比卡片
===================================  ==========================================

.. warning::

   **搜索页价格 ≠ 真实售价。** 实测同一个保温杯：``item-search`` 返回 ``44.9``，
   ``item-detail`` 返回 ``79.9``，差 78%。官方 Skill 文档也警告过同样的坑
   （爱奇艺年卡 ￥88 vs ￥135）。毛利率是权重最高的维度，用错价格整个排序就是错的 ——
   所以本模块默认 **拿不到详情就丢弃该商品**（``require_detail=True``），
   而不是退回去用搜索价。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..config import settings
from ..crawler import Source
from ..enrich import PROVENANCE_SEP
from ..models import ProductIn
from ..scoring import clamp

logger = logging.getLogger(__name__)

#: 官方 agent card 与接口地址
DEFAULT_A2A_URL = "https://pc-taoclaw.taobao.com/a2a/itemSearch"

SKILL_SEARCH = "item-search"
SKILL_DETAIL = "item-detail"
SKILL_COMPARE = "item-compare"

ARTIFACT_SEARCH = "search-candidates"
ARTIFACT_DETAIL = "item-detail-result"
ARTIFACT_COMPARE = "a2a-compare-card"

#: item-search 支持的排序（实测可用）
SORT_OPTIONS = ("sales_desc", "price_asc", "price_desc")
#: item-detail 单批上限（官方报错信息原文：「itemIds 数量需在 1-10 个之间」）
MAX_DETAIL_IDS = 10
#: item-compare 数量区间（官方报错原文：「itemIds 数量需在 2-5 个之间」）
MIN_COMPARE_IDS, MAX_COMPARE_IDS = 2, 5

#: 销量排序窗口内，第 1 名 / 最后一名的热度代理值。
#: 不取 100 是因为它是**相对位次**而非真实销量，留出保守余量。
RANK_HEAT_TOP = 95.0
RANK_HEAT_BOTTOM = 40.0

#: 只认重量单位。**绝不匹配 ml/mL/毫升/L** —— 那是容量，混淆会把重量算错几十倍
_WEIGHT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*(kg|千克|公斤|斤|mg|毫克|g|克)(?![A-Za-z])",
    re.IGNORECASE,
)
_WEIGHT_FACTORS = {
    "kg": 1.0, "千克": 1.0, "公斤": 1.0,
    "g": 0.001, "克": 0.001,
    "mg": 0.000001, "毫克": 0.000001,
    "斤": 0.5,
}

#: itemProperties 里可能直接给出重量的键
_WEIGHT_PROPERTY_KEYS = ("重量", "净重", "毛重", "商品重量", "单件重量")

#: itemProperties 里不适合当「卖点」展示的键
_PROPERTY_NOISE = {"品牌", "型号", "颜色分类", "包装种类"}

_HIGHLIGHT_TAG = re.compile(r"<[^>]+>")


class TaobaoA2AError(RuntimeError):
    """A2A 调用异常基类。"""


class TaobaoTaskFailed(TaobaoA2AError):
    """服务端返回 TASK_STATE_FAILED（message 里通常有可读原因）。"""

    def __init__(self, state: str, message: str = "") -> None:
        self.state = state
        self.message = message
        super().__init__(f"A2A 任务未完成：state={state} {message}".strip())


# --------------------------------------------------------------------------- #
# 文本处理
# --------------------------------------------------------------------------- #

def strip_highlight(text: Any) -> str:
    """去掉淘宝标题里的 ``<span class=H>`` 高亮标签。"""
    return _HIGHLIGHT_TAG.sub("", str(text or "")).strip()


def extract_weight_kg(*texts: Any) -> Optional[float]:
    """从文本里抽取重量（kg）。

    只认重量单位（kg/g/mg/斤），**不会**把 ``240mL`` 这类容量当成重量 ——
    这是最容易出错的地方：容量与重量差着密度，直接混用会把物流维度算歪。

    返回首个匹配；找不到返回 ``None``（调用方应保持缺省值而不是写 0）。
    """
    for text in texts:
        if not text:
            continue
        match = _WEIGHT_PATTERN.search(str(text))
        if not match:
            continue
        value = float(match.group(1))
        factor = _WEIGHT_FACTORS.get(match.group(2).lower())
        if factor is None:
            continue
        kg = value * factor
        if 0 < kg <= 50:
            return round(kg, 4)
    return None


def rank_to_heat(rank: int, count: int,
                 top: float = RANK_HEAT_TOP, bottom: float = RANK_HEAT_BOTTOM) -> float:
    """销量排序窗口内的位次 → 热度代理值。

    .. note::
       这是**相对位次**，不是真实销量：``sort=sales_desc`` 下第 1 名比第 50 名卖得多，
       但不同关键词、不同窗口之间不可比。返回值刻意压在 40-95 区间以体现不确定性。
    """
    if count <= 1:
        return round(top, 2)
    ratio = (max(1, rank) - 1) / (count - 1)
    return round(clamp(top - ratio * (top - bottom), bottom, top), 2)


# --------------------------------------------------------------------------- #
# A2A 客户端
# --------------------------------------------------------------------------- #

@dataclass
class A2AResult:
    """一次 A2A 调用的结果。"""

    artifact: str
    data: dict[str, Any]
    task_id: str = ""
    context_id: str = ""

    def products(self) -> list[dict[str, Any]]:
        items = self.data.get("products") or self.data.get("items") or []
        return [item for item in items if isinstance(item, Mapping)]


class TaobaoA2AClient:
    """淘宝 A2A 的 JSON-RPC 客户端。

    Args:
        interval: 两次调用之间的最小间隔（秒）。这是内测中的公开接口，调用需克制。
        retries: 网络/服务端瞬时错误的重试次数。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_A2A_URL,
        *,
        timeout: int = 45,
        interval: float = 0.8,
        retries: int = 2,
        session: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.interval = max(0.0, interval)
        self.retries = max(0, retries)
        self._session = session
        self._last_call = 0.0
        self.calls = 0

    @classmethod
    def from_settings(cls, **overrides: Any) -> "TaobaoA2AClient":
        kwargs: dict[str, Any] = {
            "base_url": settings.taobao_a2a_url,
            "timeout": settings.taobao_timeout,
            "interval": settings.taobao_interval,
            "retries": settings.taobao_retries,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def agent_card(self) -> dict[str, Any]:
        """读取公开的 agent card（用于自检）。"""
        response = self.session.get(
            f"{self.base_url}/.well-known/agent.json", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _throttle(self) -> None:
        if self.interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.monotonic()

    def call(self, payload: Mapping[str, Any]) -> A2AResult:
        """发一次 ``tasks/send``，返回第一个 artifact 的数据。"""
        envelope = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "data", "data": dict(payload)}],
                }
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._throttle()
            self.calls += 1
            try:
                return self._call_once(envelope)
            except TaobaoTaskFailed:
                raise  # 业务拒绝，重试没意义
            except Exception as exc:  # noqa: BLE001 - 网络层面统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(8.0, 1.5 * (2 ** attempt)))
                    continue
        raise TaobaoA2AError(f"A2A 调用失败（已重试 {self.retries} 次）：{last_error}")

    def _call_once(self, envelope: Mapping[str, Any]) -> A2AResult:
        response = self.session.post(
            self.base_url,
            data=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        status = getattr(response, "status_code", 200)
        text = getattr(response, "text", "")
        if status >= 400:
            raise TaobaoA2AError(f"HTTP {status}：{text[:200]}")

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise TaobaoA2AError(f"响应不是合法 JSON：{text[:200]}") from exc

        if isinstance(body, Mapping) and body.get("error"):
            error = body["error"] or {}
            raise TaobaoA2AError(
                f"JSON-RPC 错误 {error.get('code')}：{error.get('message')}"
            )

        task = ((body or {}).get("result") or {}).get("task") or {}
        state = str((task.get("status") or {}).get("state") or "")
        if state and state != "TASK_STATE_COMPLETED":
            raise TaobaoTaskFailed(state, _status_text(task))

        artifacts = task.get("artifacts") or []
        if not artifacts:
            raise TaobaoTaskFailed(state or "NO_ARTIFACT", "任务完成但没有返回 artifact")

        artifact = artifacts[0] or {}
        return A2AResult(
            artifact=str(artifact.get("name") or ""),
            data=_artifact_data(artifact),
            task_id=str(task.get("id") or ""),
            context_id=str(task.get("contextId") or ""),
        )

    # ---- 三个 skill 的封装 ----

    def search(self, query: str, *, sort: Optional[str] = None,
               limit: Optional[int] = None) -> A2AResult:
        """``item-search``：按关键词召回候选商品（仅召回，不做推荐）。"""
        payload: dict[str, Any] = {"skillId": SKILL_SEARCH, "query": query}
        if sort:
            payload["sort"] = sort
        if limit:
            payload["limit"] = int(limit)
        return self.call(payload)

    def detail(self, item_ids: Sequence[str]) -> A2AResult:
        """``item-detail``：批量取结构化详情（真实价格、规格、属性、物流）。"""
        ids = [str(item) for item in item_ids if str(item).strip()]
        if not ids:
            raise ValueError("itemIds 不能为空")
        if len(ids) > MAX_DETAIL_IDS:
            raise ValueError(f"item-detail 单批最多 {MAX_DETAIL_IDS} 个，收到 {len(ids)} 个")
        return self.call({"skillId": SKILL_DETAIL, "itemIds": ids})

    def compare(self, item_ids: Sequence[str], query: str = "") -> A2AResult:
        """``item-compare``：对 2-5 个商品做结构化对比。"""
        ids = [str(item) for item in item_ids if str(item).strip()]
        if not (MIN_COMPARE_IDS <= len(ids) <= MAX_COMPARE_IDS):
            raise ValueError(
                f"item-compare 需要 {MIN_COMPARE_IDS}-{MAX_COMPARE_IDS} 个商品，收到 {len(ids)} 个"
            )
        payload: dict[str, Any] = {"skillId": SKILL_COMPARE, "itemIds": ids}
        if query:
            payload["query"] = query
        return self.call(payload)


def _status_text(task: Mapping[str, Any]) -> str:
    """把失败任务的 message.parts 拼成一行可读文本。"""
    message = (task.get("status") or {}).get("message") or {}
    parts = message.get("parts") or []
    texts = []
    for part in parts:
        if isinstance(part, Mapping) and part.get("text"):
            texts.append(str(part["text"]))
    return " / ".join(texts)[:300]


def _artifact_data(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """取 artifact 第一个 part 的 ``data``。

    测到过 ``item-compare`` 的 part 没有 ``data``（只有渲染卡片），
    这时返回空 dict 而不是炸掉 —— 调用方按需要判断。
    """
    parts = artifact.get("parts") or []
    for part in parts:
        if isinstance(part, Mapping) and isinstance(part.get("data"), Mapping):
            data = part["data"]
            # 服务端有时会再包一层 {"result":..., "data":{...}}
            if isinstance(data.get("data"), Mapping):
                return dict(data["data"])
            return dict(data)
    return {}


# --------------------------------------------------------------------------- #
# 映射
# --------------------------------------------------------------------------- #

def to_product(
    search_item: Mapping[str, Any],
    detail: Mapping[str, Any] | None,
    *,
    query: str = "",
    rank: int = 1,
    window: int = 1,
    category: str = "",
    heat_mode: str = "rank",
    extract_weight: bool = True,
    source: str = "taobao",
) -> ProductIn:
    """把一条淘宝商品映射成 ``ProductIn``。

    Args:
        search_item: ``item-search`` 里的候选（提供 itemId / 店铺 / 发货地 / 图标）。
        detail: 对应的 ``item-detail`` 结果。**为 None 时价格只能来自搜索页**，
            调用方应确保 ``require_detail=True`` 时不会走到这里。
        heat_mode: ``rank`` 用销量排序位次做热度代理；``neutral`` 一律给 50。
    """
    item_id = str((detail or {}).get("itemId") or search_item.get("itemId") or "").strip()
    title = strip_highlight(
        (detail or {}).get("title") or search_item.get("title") or ""
    ) or f"淘宝商品{item_id}"

    product: dict[str, Any] = dict(search_item)
    if detail:
        product.update({k: v for k, v in detail.items() if v is not None})

    price = _to_float(product.get("price"))
    if price is None or price <= 0:
        price = 0.0

    shop_name = product.get("shopName") or ""
    brand = product.get("brandName") or ""
    procity = product.get("procity") or product.get("deliveryAddress") or ""

    properties = detail.get("itemProperties") if isinstance(detail, Mapping) else None
    properties = properties if isinstance(properties, Mapping) else {}

    weight: Optional[float] = None
    if extract_weight:
        for key in _WEIGHT_PROPERTY_KEYS:
            if properties.get(key):
                weight = extract_weight_kg(properties[key])
                if weight:
                    break
        if weight is None:
            weight = extract_weight_kg(
                detail.get("defaultSkuName") if isinstance(detail, Mapping) else None,
                " ".join(str(name) for name in (detail.get("skuNames") or []))
                if isinstance(detail, Mapping) else None,
                title,
            )
    weight_source = "从 SKU 名/标题抽取" if weight else "接口未提供，取缺省值"

    heat = 50.0 if heat_mode == "neutral" else rank_to_heat(rank, window)

    # 拿不到详情时价格只能来自搜索页 —— 必须写进来源说明，不能静默充当真实售价
    has_detail = bool(detail)

    note_parts = [f"淘宝A2A｜商品ID {item_id}"]
    if shop_name:
        note_parts.append(f"店铺 {shop_name}")
    if brand:
        note_parts.append(f"品牌 {brand}")
    if procity:
        note_parts.append(f"发货地 {procity}")
    highlights = [
        f"{key}={value}" for key, value in properties.items()
        if key not in _PROPERTY_NOISE and value
    ][:3]
    if highlights:
        note_parts.append("规格 " + "、".join(highlights))
    if product.get("subTitle"):
        note_parts.append(str(product["subTitle"]))
    note = " ".join(note_parts)

    provenance = (
        ("价格取 item-detail 真实售价；" if has_detail
         else "⚠️ 价格来自搜索页（通常只是起步价，比真实售价低很多），需人工复核；")
        + (f"热度按销量排序位次代理（第 {rank}/{window} 位）；" if heat_mode != "neutral"
           else "热度未提供，取中性值 50；")
        + f"重量{weight_source}；"
        + "竞争度、复购、合规、传播力接口未提供"
    )

    return ProductIn(
        title=title[:200],
        external_id=item_id[:64],
        category=(category or query or "淘宝")[:64],
        price=round(price, 2),
        cost=0.0,  # 淘宝是零售价，成本需另有来源（1688 供货价 / 表格导入）
        source=source,
        url=str(product.get("url") or product.get("auctionURL") or "")[:500],
        heat=heat,
        competition=50.0,
        weight_kg=weight if weight else 0.5,
        repurchase=50.0,
        compliance_risk=20.0,
        virality=50.0,
        note=f"{note}{PROVENANCE_SEP}{provenance}"[:1000],
    )


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

@dataclass
class TaobaoFetchReport:
    """拉取过程明细，供 CLI / UI 展示。"""

    queries: list[str] = field(default_factory=list)
    #: 搜索结果总条数（含广告、含跳关键词重复）
    recalled: int = 0
    dropped_ads: int = 0
    duplicates: int = 0
    #: 去重 + 去广告后，真正去取详情的数量
    candidates: int = 0
    detail_requested: int = 0
    detail_ok: int = 0
    dropped_no_detail: int = 0
    #: 最终产出的商品数
    kept: int = 0
    api_calls: int = 0
    warnings: list[str] = field(default_factory=list)
    #: 不是错误、但会影响打分结论的提醒
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"召回 {self.recalled} 个"]
        if self.dropped_ads:
            parts.append(f"过滤广告 {self.dropped_ads} 个")
        if self.duplicates:
            parts.append(f"去重 {self.duplicates} 个")
        parts.append(f"详情成功 {self.detail_ok}/{self.detail_requested}")
        if self.dropped_no_detail:
            parts.append(f"因无详情丢弃 {self.dropped_no_detail} 个")
        parts.append(f"入库 {self.kept} 个")
        parts.append(f"A2A 调用 {self.api_calls} 次")
        return "；".join(parts) + "。"


class TaobaoSource(Source):
    """淘宝 A2A 数据源。

    Args:
        queries: 搜索关键词列表（每个词就是一次召回）。
        limit: 每个关键词召回多少候选。
        sort: ``sales_desc`` / ``price_asc`` / ``price_desc``。
        require_detail: **默认 True** —— 拿不到 ``item-detail`` 的商品直接丢弃。
            因为搜索页价格与真实售价差过 78%，用它算毛利率会得出错误结论。
        drop_ads: 是否过滤 ``isAd=true`` 的广告位。
        max_detail: 最多为多少个商品取详情（控制调用次数）；``0`` 表示不限制。
        category: 覆盖类目名；不传则用搜索词。
    """

    name = "taobao"

    def __init__(
        self,
        queries: Optional[Iterable[str]] = None,
        *,
        limit: int = 30,
        sort: str = "sales_desc",
        detail_batch: Optional[int] = None,
        require_detail: bool = True,
        drop_ads: bool = True,
        max_detail: int = 0,
        heat_mode: str = "rank",
        extract_weight: bool = True,
        category: str = "",
        client: Optional[TaobaoA2AClient] = None,
        source_label: str = "taobao",
    ) -> None:
        self.queries = [str(q).strip() for q in (queries or []) if str(q).strip()]
        self.limit = max(1, int(limit))
        self.sort = sort if sort in SORT_OPTIONS else "sales_desc"
        self.detail_batch = min(int(detail_batch or settings.taobao_detail_batch),
                                MAX_DETAIL_IDS)
        self.require_detail = require_detail
        self.drop_ads = drop_ads
        self.max_detail = max(0, int(max_detail))
        self.heat_mode = heat_mode if heat_mode in {"rank", "neutral"} else "rank"
        self.extract_weight = extract_weight
        self.category = category
        self.source_label = source_label
        self._client = client
        self.report = TaobaoFetchReport(queries=list(self.queries))

    @property
    def client(self) -> TaobaoA2AClient:
        if self._client is None:
            self._client = TaobaoA2AClient.from_settings()
        return self._client

    def fetch(self) -> list[ProductIn]:
        if not self.queries:
            raise ValueError("淘宝数据源至少需要一个搜索关键词")

        # 1) 召回 + 去重
        candidates: dict[str, tuple[dict[str, Any], str]] = {}
        for query in self.queries:
            result = self.client.search(query, sort=self.sort, limit=self.limit)
            products = result.products()
            for item in products:
                item_id = str(item.get("itemId") or "").strip()
                if not item_id:
                    continue
                self.report.recalled += 1
                if item_id in candidates:
                    self.report.duplicates += 1
                    continue
                if self.drop_ads and item.get("isAd"):
                    self.report.dropped_ads += 1
                    continue
                candidates[item_id] = (dict(item), query)
                self.report.candidates += 1

        if not candidates:
            self.report.warnings.append("没有召回到任何候选商品（关键词可能太窄）。")
            return []

        # 2) 取详情（真实价格在这里）
        order = list(candidates)
        if self.max_detail:
            order = order[: self.max_detail]
        details = self._fetch_details(order)

        # 3) 映射
        window = len(order)
        products: list[ProductIn] = []
        for rank, item_id in enumerate(order, start=1):
            search_item, query = candidates[item_id]
            detail = details.get(item_id)
            if not detail:
                if self.require_detail:
                    self.report.dropped_no_detail += 1
                    continue
            products.append(to_product(
                search_item, detail,
                query=query, rank=rank, window=window,
                category=self.category,
                heat_mode=self.heat_mode,
                extract_weight=self.extract_weight,
                source=self.source_label,
            ))

        self.report.kept = len(products)
        self.report.api_calls = self.client.calls

        # 淘宝只有零售价、没成本 —— 不提醒的话毛利率维度会静默饱和
        if products and all(item.cost <= 0 for item in products):
            self.report.notes.append(
                "淘宝只提供零售价、没有成本，所有商品毛利率都是 100%，"
                "而毛利率是权重最高的维度 —— 它实际已饱和、不参与区分。"
                "目前排名由热度/重量等维度驱动；建议用 1688 供货价或表格导入补齐成本。"
            )

        if self.report.dropped_no_detail:
            self.report.warnings.append(
                f"{self.report.dropped_no_detail} 个商品拿不到 item-detail，"
                "已丢弃（搜索页价格与真实售价差过 78%，不能拿来算毛利率）。"
            )
        return products

    def _fetch_details(self, item_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """按批取详情；单批失败不影响其他批次。"""
        found: dict[str, dict[str, Any]] = {}
        batch_size = max(1, self.detail_batch)
        for start in range(0, len(item_ids), batch_size):
            chunk = list(item_ids[start:start + batch_size])
            self.report.detail_requested += len(chunk)
            try:
                result = self.client.detail(chunk)
            except (TaobaoA2AError, ValueError) as exc:
                logger.warning("淘宝 item-detail 批次失败（%d 个）：%s", len(chunk), exc)
                self.report.warnings.append(
                    f"item-detail 批次失败（第 {start // batch_size + 1} 批，"
                    f"{len(chunk)} 个）：{exc}"
                )
                continue
            for item in result.products():
                item_id = str(item.get("itemId") or "").strip()
                if item_id:
                    found[item_id] = dict(item)
            self.report.detail_ok += len([i for i in chunk if i in found])
        return found
