"""抖音开放平台（抖店 / 精选联盟）选品数据源。

实现完全依据官方文档，未做任何猜测：

* API 调用指南与签名算法 —— https://op.jinritemai.com/docs/guide-docs/148/814
* 选品商品搜索接口        —— https://op.jinritemai.com/docs/api-docs/61/1725
  （``buyin.kolMaterialsProductsSearch``）
* 签名工具与 access_token —— https://op.jinritemai.com/docs/guide-docs/97/1896

签名算法（官方文档第六节原文步骤，MD5 与 hmac-sha256 通用）：

1. ``param_json`` 内的键按字母升序排序，生成**紧凑** JSON（分隔符后不带空格）；
   ``&``→``\\u0026``、``<``→``\\u003c``、``>``→``\\u003e``、``\\b``→``\\u0008``。
2. 所有请求参数按字母升序排列，**``access_token`` 与 ``sign_method`` 不参与加密**。
3. 拼接为 ``key1value1key2value2...``。
4. 把 ``app_secret`` 拼接在该字符串的**两端**。
5. 对该字符串做 MD5 或 HMAC-SHA256（HMAC 的密钥同样是 ``app_secret``），取小写十六进制。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional

from ..config import settings
from ..crawler import Source
from ..models import ProductIn
from ..scoring import clamp, linear_map

logger = logging.getLogger(__name__)

#: 抖店开放平台正式环境网关
DEFAULT_BASE_URL = "https://openapi-fxg.jinritemai.com"

#: 时间戳时区固定为 GMT+8
TZ_GMT8 = timezone(timedelta(hours=8))

#: 参与签名的公共参数之外需要排除的字段
SIGN_EXCLUDED_FIELDS = frozenset({"sign", "access_token", "sign_method"})

#: 支持的成功返回码。抖店不同业务线成功码不统一：老接口用 0，精选联盟用 10000
SUCCESS_CODES = frozenset({0, 10000})

#: 官方返回码释义（docs/guide-docs/148/814 第八节）
ERROR_CODES: dict[int, str] = {
    1: "请登录后再操作",
    2: "无权限",
    3: "缺少参数",
    4: "参数错误",
    5: "参数不合法",
    6: "业务参数json解析失败，所有参数需为 string 类型",
    7: "服务器错误",
    8: "服务繁忙",
    9: "访问太频繁",
    10: "需要用 POST 请求",
    11: "签名校验失败",
    12: "版本太旧，请升级",
    302: "找不到 user_id",
    30001: "认证失败（app_key 格式不正确 / app_key 不存在 / access_token 为空）",
    30002: "access_token 已过期",
    30003: "店铺授权已失效，请重新引导商家完成店铺授权",
    30004: "应用已被系统禁用",
    30005: "access_token 不存在，请使用最新的 access_token 访问",
    30006: "店铺授权已被关闭，请联系商家打开授权开关",
    30007: "app_key 和 access_token 不匹配，请仔细检查",
    20000: "系统错误，请稍后再试",
    40004: "非法的参数",
    50002: "业务处理失败",
}

#: 接口允许的最大分页大小（文档：page_size 小于等于 20）
MAX_PAGE_SIZE = 20

#: 选品广场商品搜索接口
M_SEARCH_PRODUCTS = "buyin.kolMaterialsProductsSearch"
#: 换取 access_token 接口
M_TOKEN_CREATE = "token.create"
#: 刷新 access_token 接口
M_TOKEN_REFRESH = "token.refresh"

# ---- 维度推导参数（灵感来自接口返回字段，属于启发式映射，非官方口径）----

#: 历史销量达到该值，需求热度记 100
SALES_FOR_MAX_HEAT = 1_000_000
#: 同类在售商品数达到该值，竞争度记 100
PRODUCTS_FOR_MAX_COMPETITION = 1_000_000
#: 达人佣金率达到该值，传播潜力记 100
COMMISSION_FOR_MAX_VIRALITY = 0.30

#: 字段来源说明，供 README / 前端展示数据可信度
FIELD_PROVENANCE: dict[str, str] = {
    "heat": "接口 sales（历史总销量）取对数映射",
    "competition": "接口 total（同条件下在售商品数）取对数映射，需配合 title/类目筛选才有意义",
    "margin": "接口 kol_cos_fee / kol_cos_ratio（达人佣金）换算",
    "virality": "接口 kol_cos_ratio 代理指标（佣金越高越易撬动达人内容）",
    "weight_kg": "接口未提供，缺省 0.5，需人工复核",
    "repurchase": "接口未提供，缺省 50，需人工复核",
    "compliance_risk": "接口未提供，缺省 20，需人工复核",
}


class DouyinError(RuntimeError):
    """抖音数据源异常基类。"""


class DouyinConfigError(DouyinError):
    """缺少必要配置。"""


class DouyinAPIError(DouyinError):
    """接口返回了非成功码。"""

    def __init__(self, code: int, msg: str = "", sub_code: str = "",
                 sub_msg: str = "", detail: str = "") -> None:
        self.code = code
        self.msg = msg or ERROR_CODES.get(code, "")
        self.sub_code = sub_code
        self.sub_msg = sub_msg
        self.detail = detail
        parts = [f"code={code}", f"msg={self.msg!r}"]
        if sub_code or sub_msg:
            parts.append(f"sub=({sub_code},{sub_msg})")
        if detail:
            parts.append(detail)
        super().__init__("抖店接口调用失败：" + " ".join(parts))

    @property
    def expired_token(self) -> bool:
        """是否为 token 失效类错误（可提示重新授权）。"""
        return self.code in {30002, 30003, 30005, 30006, 30007}


# --------------------------------------------------------------------------- #
# 签名
# --------------------------------------------------------------------------- #

def _escape_param_json(text: str) -> str:
    """按官方要求转义 param_json 中的特殊字符。"""
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\\b", "\\u0008")
    )


def _normalize_value(value: Any) -> Any:
    """参数值统一转为字符串（官方：param_json 中的所有参数都需为 string 类型）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return str(value)


def canonical_param_json(params: Mapping[str, Any] | None) -> str:
    """生成用于签名的 param_json：键升序、紧凑分隔符、特殊字符转义。"""
    cleaned = {
        key: _normalize_value(value)
        for key, value in (params or {}).items()
        if value is not None
    }
    raw = json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _escape_param_json(raw)


def build_sign_string(public_params: Mapping[str, str], app_secret: str) -> str:
    """构造待加密字符串：``app_secret`` + 排序拼接的 k/v + ``app_secret``。

    ``access_token`` / ``sign_method`` / ``sign`` 不参与。
    """
    items = sorted(
        (key, value)
        for key, value in public_params.items()
        if key not in SIGN_EXCLUDED_FIELDS
    )
    joined = "".join(f"{key}{value}" for key, value in items)
    return f"{app_secret}{joined}{app_secret}"


def sign_params(public_params: Mapping[str, str], app_secret: str,
                sign_method: str = "hmac-sha256") -> str:
    """计算 sign，返回小写十六进制字符串。"""
    message = build_sign_string(public_params, app_secret).encode("utf-8")
    method = (sign_method or "hmac-sha256").lower()
    if method == "md5":
        return hashlib.md5(message).hexdigest()
    if method in {"hmac-sha256", "hmac_sha_256"}:
        return hmac.new(
            app_secret.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()
    raise DouyinConfigError(f"不支持的签名算法：{sign_method!r}，可选 md5 / hmac-sha256")


def method_to_path(method: str) -> str:
    """``buyin.kolMaterialsProductsSearch`` → ``/buyin/kolMaterialsProductsSearch``。"""
    return "/" + method.replace(".", "/")


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #

class DouyinClient:
    """抖店开放平台 HTTP 客户端。"""

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        access_token: str = "",
        base_url: str = DEFAULT_BASE_URL,
        sign_method: str = "hmac-sha256",
        timeout: int = 30,
        shop_id: str = "",
        session: Any = None,
    ) -> None:
        if not app_key or not app_secret:
            raise DouyinConfigError(
                "缺少 app_key / app_secret，请在 .env 配置 APS_DOUYIN_APP_KEY 与 "
                "APS_DOUYIN_APP_SECRET"
            )
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = access_token
        self.base_url = base_url.rstrip("/")
        self.sign_method = sign_method
        self.timeout = timeout
        self.shop_id = shop_id
        self._session = session

    @classmethod
    def from_settings(cls, **overrides: Any) -> "DouyinClient":
        """从全局配置构造客户端。"""
        kwargs: dict[str, Any] = {
            "app_key": settings.douyin_app_key,
            "app_secret": settings.douyin_app_secret,
            "access_token": settings.douyin_access_token,
            "base_url": settings.douyin_base_url,
            "sign_method": settings.douyin_sign_method,
            "timeout": settings.douyin_timeout,
            "shop_id": settings.douyin_shop_id,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def session(self) -> Any:
        """惰性创建 requests.Session。"""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(TZ_GMT8).strftime("%Y-%m-%d %H:%M:%S")

    def build_public_params(self, method: str, param_json: str,
                            timestamp: Optional[str] = None) -> dict[str, str]:
        """组装公共参数（尚未签名）。"""
        params = {
            "method": method,
            "app_key": self.app_key,
            "param_json": param_json,
            "timestamp": timestamp or self._timestamp(),
            "v": "2",
        }
        if self.access_token:
            params["access_token"] = self.access_token
        return params

    def call(self, method: str, params: Mapping[str, Any] | None = None,
             need_token: bool = True, http_method: str = "POST",
             timestamp: Optional[str] = None) -> dict[str, Any]:
        """调用任意抖店接口，返回 ``data`` 字段。

        Args:
            method: 接口名，形如 ``buyin.kolMaterialsProductsSearch``。
            params: 业务参数，会被序列化进 ``param_json``。
            need_token: 是否携带 ``access_token``（``token.create`` 不需要）。
            http_method: POST（默认，param_json 放 body）或 GET。
            timestamp: 指定时间戳，仅用于测试与重放；默认取当前 GMT+8 时间。

        Raises:
            DouyinAPIError: 返回码非成功码，或 HTTP / JSON 层面失败。
        """
        param_json = canonical_param_json(params)
        public_params = self.build_public_params(method, param_json, timestamp=timestamp)

        access_token = public_params.pop("access_token", None)
        if need_token:
            if not access_token:
                raise DouyinConfigError(
                    "缺少 access_token，请配置 APS_DOUYIN_ACCESS_TOKEN，"
                    "或先用 DouyinClient.create_self_token() 换取"
                )
            public_params["access_token"] = access_token

        public_params["sign"] = sign_params(public_params, self.app_secret, self.sign_method)
        public_params["sign_method"] = self.sign_method

        url = f"{self.base_url}{method_to_path(method)}"
        # param_json 走 body，其余公共参数走 query，避免超长 URL 与特殊字符转义问题
        query = {key: value for key, value in public_params.items() if key != "param_json"}
        headers = {"Content-Type": "application/json"}

        try:
            if http_method.upper() == "GET":
                response = self.session.get(
                    url, params={**query, "param_json": param_json},
                    headers=headers, timeout=self.timeout,
                )
            else:
                response = self.session.post(
                    url, params=query, data=param_json.encode("utf-8"),
                    headers=headers, timeout=self.timeout,
                )
        except Exception as exc:  # noqa: BLE001 - 网络异常统一包装
            raise DouyinAPIError(-1, f"请求失败：{exc}", detail=url) from exc

        status = getattr(response, "status_code", 200)
        text = getattr(response, "text", "")
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise DouyinAPIError(
                -2, f"响应不是合法 JSON（HTTP {status}）", detail=text[:300]
            ) from exc

        if not isinstance(payload, dict):
            raise DouyinAPIError(-3, "响应结构异常", detail=str(payload)[:300])

        try:
            code = int(payload.get("code", -3))
        except (TypeError, ValueError):
            raise DouyinAPIError(-3, "响应缺少合法的 code 字段", detail=str(payload)[:300])

        if code not in SUCCESS_CODES:
            raise DouyinAPIError(
                code,
                str(payload.get("msg") or ""),
                str(payload.get("sub_code") or ""),
                str(payload.get("sub_msg") or ""),
            )

        data = payload.get("data")
        return data if isinstance(data, dict) else {}

    # ---- 授权相关 ----

    def create_self_token(self, shop_id: Optional[str] = None) -> dict[str, Any]:
        """自用型应用换取 access_token（grant_type=authorization_self）。

        文档：https://op.jinritemai.com/docs/guide-docs/97/1896
        """
        sid = str(shop_id or self.shop_id or "")
        if not sid:
            raise DouyinConfigError("换取 access_token 需要 shop_id（APS_DOUYIN_SHOP_ID）")
        data = self.call(
            M_TOKEN_CREATE,
            {"code": "", "grant_type": "authorization_self", "shop_id": sid},
            need_token=False,
        )
        if data.get("access_token"):
            self.access_token = str(data["access_token"])
        return data

    def create_token_by_code(self, code: str) -> dict[str, Any]:
        """店铺授权码模式换取 access_token（grant_type=authorization_code）。"""
        if not code:
            raise DouyinConfigError("授权码不能为空")
        data = self.call(
            M_TOKEN_CREATE,
            {"code": code, "grant_type": "authorization_code"},
            need_token=False,
        )
        if data.get("access_token"):
            self.access_token = str(data["access_token"])
        return data

    def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """刷新 access_token。"""
        if not refresh_token:
            raise DouyinConfigError("refresh_token 不能为空")
        data = self.call(
            M_TOKEN_REFRESH,
            {"grant_type": "refresh_token", "refresh_token": refresh_token},
            need_token=False,
        )
        if data.get("access_token"):
            self.access_token = str(data["access_token"])
        return data

    # ---- 选品 ----

    def search_products(
        self,
        *,
        title: str = "",
        page: int = 1,
        page_size: int = MAX_PAGE_SIZE,
        search_type: int = 1,
        sort_type: int = 1,
        first_cids: Optional[Iterable[int]] = None,
        second_cids: Optional[Iterable[int]] = None,
        third_cids: Optional[Iterable[int]] = None,
        price_min: Optional[int] = None,
        price_max: Optional[int] = None,
        sell_num_min: Optional[int] = None,
        sell_num_max: Optional[int] = None,
        cos_ratio_min: Optional[int] = None,
        cos_ratio_max: Optional[int] = None,
        share_status: int = 1,
        tag: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """调用 ``buyin.kolMaterialsProductsSearch`` 搜索选品广场商品。

        Args:
            title: 标题关键词，留空则返回全量。
            search_type: 召回排序条件，0 默认 / 1 历史销量 / 2 价格 / 3 佣金金额 / 4 佣金比例。
            sort_type: 0 升序 / 1 降序。
            price_min/max: 价格区间，单位**分**。
            cos_ratio_min/max: 佣金率区间，**乘 100**（1.1% 传 110）。
            share_status: 1 仅可分销商品，0 全量。
            tag: 0 全量 / 1 超值购 / 2 抖音超市。

        Returns:
            ``{"total": int, "products": [...]}``
        """
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError(f"page_size 必须在 1~{MAX_PAGE_SIZE} 之间（官方限制）")

        params: dict[str, Any] = {
            "title": title or None,
            "first_cids": list(first_cids) if first_cids else None,
            "second_cids": list(second_cids) if second_cids else None,
            "third_cids": list(third_cids) if third_cids else None,
            "price_min": price_min,
            "price_max": price_max,
            "sell_num_min": sell_num_min,
            "sell_num_max": sell_num_max,
            "cos_ratio_min": cos_ratio_min,
            "cos_ratio_max": cos_ratio_max,
            "search_type": search_type,
            "sort_type": sort_type,
            "page": page,
            "page_size": page_size,
            "share_status": share_status,
            "tag": tag,
        }
        if extra:
            params.update(extra)

        data = self.call(M_SEARCH_PRODUCTS, params)
        products = data.get("products") or []
        if isinstance(products, dict):  # 个别接口会包一层
            products = products.get("products") or []
        return {
            "total": _to_int(data.get("total")),
            "products": list(products),
        }


# --------------------------------------------------------------------------- #
# 响应映射
# --------------------------------------------------------------------------- #

def _to_int(value: Any, default: int = 0) -> int:
    """接口所有字段都是字符串，这里做一次宽松转换。"""
    if value is None or value == "":
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def sales_to_heat(sales: int) -> float:
    """历史销量 → 需求热度（对数映射，0 销量记 0 分）。"""
    if sales <= 0:
        return 0.0
    ratio = math.log10(sales + 1) / math.log10(SALES_FOR_MAX_HEAT + 1)
    return round(clamp(ratio * 100.0), 2)


def in_sale_count_to_competition(total: int) -> float:
    """同条件在售商品数 → 竞争度（对数映射，越多越卷）。"""
    if total <= 0:
        return 0.0
    ratio = math.log10(total + 1) / math.log10(PRODUCTS_FOR_MAX_COMPETITION + 1)
    return round(clamp(ratio * 100.0), 2)


def commission_to_virality(commission_ratio: float) -> float:
    """达人佣金率 → 内容传播力代理指标（0% 记 30 分，30% 及以上记 100 分）。"""
    return round(linear_map(commission_ratio, 0.0, COMMISSION_FOR_MAX_VIRALITY, 30.0, 100.0), 2)


def to_product(raw: Mapping[str, Any], *, competition_total: int = 0,
               source: str = "douyin") -> ProductIn:
    """把选品广场返回的单条商品映射为 ``ProductIn``。

    单位换算依据官方字段说明：
    ``price`` / ``coupon_price`` / ``cos_fee`` / ``kol_cos_fee`` 单位为**分**；
    ``kol_cos_ratio`` 为百分数乘 100（例如 10.00 表示 10%）。

    ``cost`` 被反推为「售价 − 达人佣金」，因此规则引擎算出的毛利率恰好等于
    达人佣金率 —— 这是**分销带货视角**的口径。自营商家请自行覆盖 ``cost``。
    """
    price_fen = _to_int(raw.get("price"))
    kol_fee_fen = _to_int(raw.get("kol_cos_fee"))
    kol_ratio = _to_float(raw.get("kol_cos_ratio")) / 100.0
    cos_ratio = _to_float(raw.get("cos_ratio")) / 100.0

    # 佣金比例优先用达人佣金，其次普通佣金
    effective_ratio = kol_ratio or cos_ratio
    if kol_fee_fen > 0:
        cost_fen = max(0, price_fen - kol_fee_fen)
        effective_ratio = kol_fee_fen / price_fen if price_fen else 0.0
    else:
        cost_fen = int(round(price_fen * (1.0 - effective_ratio)))

    price = round(price_fen / 100.0, 2)
    cost = round(min(cost_fen, price_fen) / 100.0, 2)

    sales = _to_int(raw.get("sales"))
    product_id = str(raw.get("product_id") or "")
    shop_name = str(raw.get("shop_name") or "")
    coupon_fen = _to_int(raw.get("coupon_price"))

    note_parts = [
        f"抖音精选联盟｜商品ID {product_id or '未知'}",
        f"店铺 {shop_name or '未知'}",
        f"历史销量 {sales}",
        f"达人佣金率 {effective_ratio:.2%}",
    ]
    if coupon_fen:
        note_parts.append(f"券后价 {coupon_fen / 100:.2f} 元")
    if str(raw.get("post_free")).lower() == "true":
        note_parts.append("包邮")
    note_parts.append(
        "｜数据说明：热度/竞争度/毛利率/传播力由接口字段推导；"
        "重量、复购、合规为缺省值，需人工复核"
    )

    return ProductIn(
        title=str(raw.get("title") or "").strip() or f"抖音商品{product_id}",
        category=_category_label(raw),
        price=price,
        cost=cost,
        source=source,
        url=str(raw.get("detail_url") or ""),
        heat=sales_to_heat(sales),
        competition=in_sale_count_to_competition(competition_total),
        weight_kg=0.5,
        repurchase=50.0,
        compliance_risk=20.0,
        virality=commission_to_virality(effective_ratio),
        note=" ".join(note_parts),
    )


def _category_label(raw: Mapping[str, Any]) -> str:
    """接口只返回类目 ID，没有名称；用三级类目 ID 拼一个可读标签。"""
    third = _to_int(raw.get("third_cid"))
    second = _to_int(raw.get("second_cid"))
    first = _to_int(raw.get("first_cid"))
    if third:
        return f"抖音类目-{third}"
    if second:
        return f"抖音类目-{second}"
    if first:
        return f"抖音类目-{first}"
    return "未分类"


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

class DouyinSource(Source):
    """精选联盟选品数据源。"""

    name = "douyin"

    def __init__(
        self,
        keywords: Optional[Iterable[str]] = None,
        *,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int = 1,
        search_type: int = 1,
        sort_type: int = 1,
        first_cids: Optional[Iterable[int]] = None,
        cos_ratio_min: Optional[int] = None,
        only_in_stock: bool = True,
        client: Optional[DouyinClient] = None,
        **search_extra: Any,
    ) -> None:
        self.keywords = [k.strip() for k in (keywords or []) if str(k).strip()] or [""]
        self.page_size = min(int(page_size), MAX_PAGE_SIZE)
        self.max_pages = max(1, int(max_pages))
        self.search_type = int(search_type)
        self.sort_type = int(sort_type)
        self.first_cids = list(first_cids) if first_cids else None
        self.cos_ratio_min = cos_ratio_min
        self.only_in_stock = only_in_stock
        self.search_extra = dict(search_extra)
        self._client = client

    @property
    def client(self) -> DouyinClient:
        if self._client is None:
            self._client = DouyinClient.from_settings()
        return self._client

    def fetch(self) -> list[ProductIn]:
        """按关键词逐页拉取并映射；同一 ``product_id`` 只保留一次。"""
        collected: dict[str, ProductIn] = {}
        for keyword in self.keywords:
            for page in range(1, self.max_pages + 1):
                data = self.client.search_products(
                    title=keyword,
                    page=page,
                    page_size=self.page_size,
                    search_type=self.search_type,
                    sort_type=self.sort_type,
                    first_cids=self.first_cids,
                    cos_ratio_min=self.cos_ratio_min,
                    **self.search_extra,
                )
                products = data.get("products") or []
                if not products:
                    break

                total = int(data.get("total") or 0)
                logger.info(
                    "抖音选品：关键词=%r 第 %d 页，返回 %d 条（同条件在售 %d 件）",
                    keyword or "<全量>", page, len(products), total,
                )
                for raw in products:
                    if self.only_in_stock and str(raw.get("in_stock")) == "0":
                        continue
                    if str(raw.get("sharable")).lower() == "false":
                        continue
                    product = to_product(raw, competition_total=total)
                    key = str(raw.get("product_id") or product.title)
                    collected.setdefault(key, product)

                if len(products) < self.page_size:
                    break

        return list(collected.values())
