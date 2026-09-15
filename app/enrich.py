"""补齐精选联盟接口未提供的维度：重量、复购潜力、合规风险。

三条来源，优先级从高到低：

1. **商品详情接口** ``product.detail`` —— 仅对**自己店铺**的商品有效。
   官方对该接口 ``product_id`` 的说明是「抖店系统生成，**店铺下唯一**」，
   因此精选联盟里其他商家的商品查不到（会返回「商品不存在」）。
2. **大模型估算** —— 任意商品，按标题 + 类目推断。结果一律标注为估算值。
3. **保持缺省值** —— 并继续在 ``note`` 里标注「需人工复核」。

设计原则：**绝不把估算值伪装成接口数据**。每个被补齐的字段都会写进
``note`` 的数据来源说明，调用方也能从 :class:`EnrichReport` 拿到明细。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import llm
from .config import settings
from .models import ProductIn
from .sources.douyin import DouyinAPIError, DouyinClient, parse_weight_kg

logger = logging.getLogger(__name__)

#: note 里数据来源说明的分隔符（与 app.sources.douyin.to_product 保持一致）
PROVENANCE_SEP = "｜数据说明："

#: 需要补齐的字段 → 中文名
TARGET_FIELDS: dict[str, str] = {
    "weight_kg": "重量",
    "repurchase": "复购",
    "compliance_risk": "合规",
    "heat": "热度",
    "virality": "传播",
    "category": "类目",
}

#: 由接口字段推导的维度 → 中文名（未被大模型接管时写进数据来源说明）
API_FIELDS: dict[str, str] = {
    "heat": "热度",
    "competition": "竞争度",
    "margin": "毛利率",
    "virality": "传播力",
}

FIELD_LIMITS: dict[str, tuple[float, float]] = {
    "weight_kg": (0.01, 50.0),
    "repurchase": (0.0, 100.0),
    "compliance_risk": (0.0, 100.0),
    "heat": (0.0, 100.0),
    "virality": (0.0, 100.0),
}

#: 接口完全拿不到的字段：一律用大模型填
BASE_FIELDS: tuple[str, ...] = ("weight_kg", "repurchase", "compliance_risk")
#: 「判断类」字段：接口只有代理指标，大模型介入才有意义
JUDGE_FIELDS: tuple[str, ...] = ("heat", "virality")

FIELD_SPECS: dict[str, str] = {
    "weight_kg": (
        "- weight_kg：单件商品（含常规包装）的运输重量，单位 kg，保留 2 位小数。\n"
        "  参考量级：手机壳 0.05、T 恤 0.25、保温杯 0.35、咖啡粉 30 条 0.4、\n"
        "  折叠椅 3.2、小型家电 1.5、智能垃圾桶 2.9。"
    ),
    "repurchase": (
        "- repurchase：复购潜力 0-100。食品/日化/耗材偏高（60-90）；\n"
        "  家电/家具/耐用品偏低（5-25）。"
    ),
    "compliance_risk": (
        "- compliance_risk：合规风险 0-100。涉及医疗器械、儿童玩具、化妆品、\n"
        "  保健或疗效宣称、锂电池运输的偏高（40-80）；普通日用百货偏低（5-20）。"
    ),
    "heat": (
        "- heat：需求热度 0-100。属于**品类**层面的判断，不是单个商品的销量。\n"
        "  刚需高频（纸品、清洁、粮油）70-85；季节性/尝鲜型（新奇小家电）50-70；\n"
        "  小众垂类（专业器材、收藏品）20-40。"
    ),
    "virality": (
        "- virality：内容传播力 0-100。指该商品在短视频里做演示/测评的天然效果。\n"
        "  有视觉冲击或前后对比的偏高（清洁神器、宠物玩具 80-95）；\n"
        "  无形或平淡的偏低（数据线、保鲜袋 25-45）。"
    ),
    "category_name": (
        "- category_name：6 个汉字以内的中文类目名（如「咖啡冲调」「宠物玩具」），\n"
        "  必须比原始类目 ID 更有信息量。"
    ),
}


def build_system_prompt(fields: Sequence[str]) -> str:
    """按需生成系统提示词 —— 未开启判断类字段时不多问，省 token。"""
    specs = [FIELD_SPECS[name] for name in fields if name in FIELD_SPECS]
    example: list[str] = ['"index":0']
    for name in fields:
        if name == "category_name":
            example.append('"category_name":"咖啡冲调"')
        elif name == "weight_kg":
            example.append('"weight_kg":0.4')
        else:
            example.append(f'"{name}":60')
    example.append('"reason":"判断依据，40 字以内"')
    body = "{" + ",".join(example) + "}"

    return (
        "你是跨境电商选品与供应链专家。用户会给出商品标题与类目，请估算以下字段：\n\n"
        + "\n".join(specs)
        + f'\n\n只输出 JSON，不要 markdown 代码块，不要解释。格式：\n{{"items":[{body}]}}'
        + "\n\n硬性要求：\n"
        "1. index 必须与输入编号一致；\n"
        "2. 每个字段都必须给出具体数值，不要返回 null；\n"
        "3. reason 不超过 40 字，说明主要判断依据；\n"
        "4. 独立判断，不要为了迎合输入里已有的数值而靠拢。"
    )


#: 已废弃的固定提示词，保留供旧调用方引用
LLM_SYSTEM_PROMPT = build_system_prompt([*BASE_FIELDS])


@dataclass
class EnrichReport:
    """补齐过程的明细，供 CLI / UI 展示。"""

    detail_attempted: int = 0
    detail_filled: int = 0
    detail_failed: int = 0
    llm_requested: int = 0
    llm_filled: int = 0
    llm_cached: int = 0
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def filled(self) -> int:
        """被补齐过至少一个字段的商品数量（粗略统计，按字段累加）。"""
        return self.detail_filled + self.llm_filled

    def summary(self) -> str:
        parts = []
        if self.detail_attempted:
            parts.append(
                f"详情接口：尝试 {self.detail_attempted} 个，成功 {self.detail_filled} 个，"
                f"失败 {self.detail_failed} 个"
            )
        if self.llm_requested:
            parts.append(
                f"大模型估算：请求 {self.llm_requested} 个，成功 {self.llm_filled} 个"
                + (f"（命中缓存 {self.llm_cached} 个）" if self.llm_cached else "")
            )
        if not parts:
            return "未执行任何补齐操作。"
        return "；".join(parts) + "。"


# --------------------------------------------------------------------------- #
# 大模型估算结果缓存（估算要花 token，同一商品不必重复问）
# --------------------------------------------------------------------------- #

class EstimateCache:
    """按 ``标题 + 类目`` 做键的 JSON 文件缓存。"""

    def __init__(self, path: Path | str | None = None, *, enabled: bool = True) -> None:
        self.path = Path(path) if path else settings.enrich_cache_path
        self.enabled = enabled
        self._data: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if self.enabled:
            self._load()

    @staticmethod
    def make_key(product: ProductIn) -> str:
        raw = f"{product.title}|{product.category}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._data = {
                    str(key): value
                    for key, value in payload.items()
                    if isinstance(value, dict)
                }
        except Exception as exc:  # noqa: BLE001 - 缓存损坏不应影响主流程
            logger.warning("估算缓存读取失败，将忽略：%s", exc)
            self._data = {}

    def get(self, product: ProductIn) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        return self._data.get(self.make_key(product))

    def put(self, product: ProductIn, estimate: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        self._data[self.make_key(product)] = dict(estimate)
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._dirty = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("估算缓存写入失败：%s", exc)

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def _coerce(value: Any, low: float, high: float) -> Optional[float]:
    """把模型返回值转成受限浮点数；无效返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(max(low, min(high, number)), 4)


def note_body(note: str) -> str:
    """去掉 note 尾部的数据来源说明。"""
    return (note or "").split(PROVENANCE_SEP)[0]


def describe_sources(sources: Mapping[str, str]) -> str:
    """生成 note 尾部的数据来源说明。"""
    from_api = [label for key, label in API_FIELDS.items() if key not in sources]
    head = ("、".join(from_api) + "由接口字段推导") if from_api else "无接口推导维度"
    if not sources:
        return f"{head}；重量、复购、合规为缺省值，需人工复核"
    bits = [f"{TARGET_FIELDS.get(key, key)}={value}" for key, value in sources.items()]
    return f"{head}；" + "、".join(bits)


def with_sources(note: str, sources: Mapping[str, str]) -> str:
    """把数据来源说明写回 note。"""
    return f"{note_body(note)}{PROVENANCE_SEP}{describe_sources(sources)}"


# --------------------------------------------------------------------------- #
# 路径 1：商品详情接口（仅自己店铺的商品）
# --------------------------------------------------------------------------- #

def fill_weight_from_detail(
    products: Sequence[ProductIn],
    client: DouyinClient,
    *,
    limit: int = 0,
    report: Optional[EnrichReport] = None,
    sources: Optional[dict[int, dict[str, str]]] = None,
) -> list[ProductIn]:
    """用 ``product.detail`` 补齐重量。

    只对**已授权店铺自己的商品**有效。精选联盟商品的 ``product_id`` 属于其他商家，
    调用会返回「商品不存在」，这类失败会被计入 ``report.detail_failed`` 而不中断流程。

    Args:
        limit: 最多尝试多少个商品（每个商品 1 次接口调用）。``0`` 表示全部。
    """
    report = report if report is not None else EnrichReport()
    sources = sources if sources is not None else {}
    items = list(products)

    candidates = [i for i, item in enumerate(items) if _extract_product_id(item)]
    if limit:
        candidates = candidates[:limit]

    for index in candidates:
        item = items[index]
        product_id = _extract_product_id(item)
        if not product_id:
            continue
        report.detail_attempted += 1
        try:
            detail = client.get_product_detail(product_id=product_id)
        except DouyinAPIError as exc:
            report.detail_failed += 1
            if len(report.errors) < 5:
                report.errors.append(f"{item.title[:20]}：{exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            report.detail_failed += 1
            if len(report.errors) < 5:
                report.errors.append(f"{item.title[:20]}：{exc}")
            continue

        weight = parse_weight_kg(detail)
        if weight:
            items[index] = item.model_copy(update={"weight_kg": weight})
            sources.setdefault(index, {})["weight_kg"] = "商品详情接口"
            report.detail_filled += 1

    if report.detail_attempted and not report.detail_filled:
        report.notes.append(
            "商品详情接口未补到任何重量。该接口只能查已授权店铺自己的商品，"
            "精选联盟里其他商家的商品查不到（会返回「商品不存在」）。"
        )
    return items


def _extract_product_id(product: ProductIn) -> Optional[str]:
    """取来源平台的商品 ID。

    优先用显式字段 ``external_id``；旧数据（该字段引入前导入的）回退到解析 note，
    以便升级后无需重新导入。
    """
    if product.external_id:
        return product.external_id.strip() or None

    marker = "商品ID "
    note = product.note or ""
    if marker not in note:
        return None
    tail = note.split(marker, 1)[1]
    token = tail.split(" ", 1)[0].strip()
    return token or None


# --------------------------------------------------------------------------- #
# 路径 2：大模型估算
# --------------------------------------------------------------------------- #

def _build_user_prompt(batch: Sequence[ProductIn]) -> str:
    lines = []
    for index, item in enumerate(batch):
        lines.append(f'{index}. 标题：{item.title} ｜ 类目：{item.category}')
    return "\n".join(lines)


#: 每个字段的落库策略
#: fill       —— 接口没有这个数据，直接填
#: fill_only  —— 接口有更可靠的硬数据（如销量），只在缺失时填
#: override   —— 接口只有代理指标，默认用大模型判断替代
VALUE_POLICY: dict[str, str] = {
    "weight_kg": "fill",
    "repurchase": "fill",
    "compliance_risk": "fill",
    "virality": "override",
    "heat": "fill_only",
}

#: 形如「抖音类目-2634」的占位类目，才允许被大模型给出的名称替换
PLACEHOLDER_CATEGORY = re.compile(r"^(抖音类目-\d+|未分类)?$")


def _clean_category_name(value: Any) -> Optional[str]:
    """清洗模型给出的类目名。"""
    if value is None:
        return None
    text = str(value).strip().replace("\n", " ")[:32]
    if not text or text.lower() in {"null", "none", "unknown", "未知"}:
        return None
    return text


def _parse_estimates(payload: Any, size: int) -> dict[int, dict[str, Any]]:
    """把模型返回解析为 {index: {field: value}}。"""
    items: Any = payload
    if isinstance(payload, Mapping):
        items = payload.get("items") or payload.get("data") or []
    if not isinstance(items, list):
        return {}

    parsed: dict[int, dict[str, Any]] = {}
    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < size:
            continue
        values: dict[str, Any] = {}
        for name, (low, high) in FIELD_LIMITS.items():
            number = _coerce(entry.get(name), low, high)
            if number is not None:
                values[name] = number
        category = _clean_category_name(entry.get("category_name"))
        if category:
            values["category_name"] = category
        if values:
            parsed[index] = values
    return parsed


def _apply(
    product: ProductIn,
    values: Mapping[str, Any],
    *,
    allowed: Optional[set[str]] = None,
    override_heat: bool = False,
) -> tuple[ProductIn, dict[str, str]]:
    """按字段策略把估算值写回商品。

    Args:
        allowed: 本次实际请求过的字段集合。**只会应用集合内的字段** ——
            避免模型多吐了未开启的字段（如未开 judge 却返回 virality）就被采纳。
            ``None`` 表示不限制。

    Returns:
        ``(新商品, {字段: 来源标签})``；无任何变更时标签为空字典。
    """
    updates: dict[str, Any] = {}
    labels: dict[str, str] = {}

    for field in ("weight_kg", "repurchase", "compliance_risk", "virality", "heat"):
        if field not in values:
            continue
        if allowed is not None and field not in allowed:
            continue  # 本次没问这个字段，即使模型返回了也不采纳
        policy = VALUE_POLICY[field]
        if field == "heat" and override_heat:
            policy = "override"
        current = getattr(product, field, 0.0) or 0.0

        if policy == "override":
            if field == "heat":
                labels[field] = f"大模型覆盖（原销量映射 {current:.0f}）"
            else:
                labels[field] = f"大模型判断（原佣金率代理 {current:.0f}）"
            updates[field] = values[field]
        elif policy == "fill_only":
            if current > 0:
                continue  # 接口有硬数据，不覆盖
            labels[field] = "大模型估算（接口值为 0）"
            updates[field] = values[field]
        else:  # fill
            labels[field] = "大模型估算"
            updates[field] = values[field]

    if allowed is None or "category_name" in allowed:
        name = _clean_category_name(values.get("category_name"))
        if name and PLACEHOLDER_CATEGORY.match(product.category or ""):
            updates["category"] = name
            labels["category"] = f"大模型判断（原 {product.category or '空'}）"

    if not updates:
        return product, {}
    return product.model_copy(update=updates), labels


def estimate_with_llm(
    products: Sequence[ProductIn],
    *,
    batch_size: Optional[int] = None,
    cache: Optional[EstimateCache] = None,
    report: Optional[EnrichReport] = None,
    sources: Optional[dict[int, dict[str, str]]] = None,
    judge: bool = False,
    override_heat: bool = False,
) -> list[ProductIn]:
    """用大模型估算接口拿不到、或只有弱代理指标的维度。

    Args:
        judge: 是否让大模型接管「判断类」字段。开启后：
            ``virality`` 用大模型判断替代佣金率代理；
            ``heat`` 只在接口推导值为 0 时填充（除非 ``override_heat=True``）；
            并顺带给出可读中文类目名，替换「抖音类目-2634」这类占位值。
        override_heat: 是否允许大模型**覆盖**由销量推导出的需求热度。
            默认关闭——接口的 ``sales`` 是硬数据，大模型判断通常不如它可靠。

    未配置 LLM 或调用失败时原样返回，不影响主流程。
    """
    report = report if report is not None else EnrichReport()
    sources = sources if sources is not None else {}
    items = list(products)

    if not items:
        return items
    if not llm.is_available():
        report.notes.append("未配置 APS_LLM_* ，跳过维度估算，保持接口值或缺省值。")
        return items

    fields: list[str] = list(BASE_FIELDS)
    if judge:
        fields.extend(JUDGE_FIELDS)
        fields.append("category_name")
    system_prompt = build_system_prompt(fields)

    batch_size = batch_size or settings.enrich_batch_size
    batch_size = max(1, min(int(batch_size), 20))
    cache = cache if cache is not None else EstimateCache()

    pending: list[int] = []
    for index, item in enumerate(items):
        cached = cache.get(item)
        if cached and judge and not set(JUDGE_FIELDS).issubset(cached):
            cached = None  # 缓存来自未开启判断的旧运行，重新问
        if not cached:
            pending.append(index)
            continue

        report.llm_cached += 1
        values = _parse_estimates({"items": [dict(cached, index=0)]}, 1).get(0, {})
        if not values:
            continue
        updated, labels = _apply(
            item, values, allowed=set(fields), override_heat=override_heat
        )
        if labels:
            items[index] = updated
            sources.setdefault(index, {}).update(labels)
            report.llm_filled += 1

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        batch = [items[i] for i in chunk]
        content = llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_user_prompt(batch)},
            ],
            temperature=0.2,
        )
        report.llm_requested += len(chunk)
        if not content:
            report.errors.append(f"大模型无返回（第 {start // batch_size + 1} 批，{len(chunk)} 条）")
            continue

        parsed = _parse_estimates(llm.extract_json(content), len(chunk))
        if not parsed:
            report.errors.append(
                f"大模型返回无法解析（第 {start // batch_size + 1} 批）：{content[:120]}"
            )
            continue

        for offset, values in parsed.items():
            index = chunk[offset]
            updated, labels = _apply(
                items[index], values, allowed=set(fields), override_heat=override_heat
            )
            if labels:
                items[index] = updated
                sources.setdefault(index, {}).update(labels)
                report.llm_filled += 1
            cache.put(items[index], {"index": 0, **values})

    cache.save()
    if report.llm_filled:
        extra = "热度/传播力/类目为大模型判断值；" if judge else ""
        report.notes.append(
            "重量/复购/合规为大模型估算值；" + extra
            + "均为估算，非接口数据，已写入 note 的数据来源说明。"
        )
    return items


# --------------------------------------------------------------------------- #
# 编排入口
# --------------------------------------------------------------------------- #

def enrich(
    products: Iterable[ProductIn],
    *,
    client: Optional[DouyinClient] = None,
    use_llm: bool = True,
    detail_limit: int = 0,
    batch_size: Optional[int] = None,
    cache: Optional[EstimateCache] = None,
    use_cache: bool = True,
    judge: bool = False,
    override_heat: bool = False,
) -> tuple[list[ProductIn], EnrichReport]:
    """按优先级补齐维度，并重写 note 里的数据来源说明。

    Args:
        client: 传入 ``DouyinClient`` 才会尝试走商品详情接口（仅自己店铺商品有效）。
        use_llm: 是否用大模型估算。
        detail_limit: 商品详情接口最多尝试多少个商品；``0`` 表示全部。
        use_cache: 是否使用估算结果缓存。
        judge: 让大模型接管传播力/类目，并在热度为 0 时填充热度。
        override_heat: 允许大模型覆盖接口由销量推导出的需求热度（默认关闭）。

    Returns:
        ``(补齐后的商品列表, 明细报告)``
    """
    report = EnrichReport()
    items = list(products)
    sources: dict[int, dict[str, str]] = {index: {} for index in range(len(items))}

    if client is not None:
        items = fill_weight_from_detail(
            items, client, limit=detail_limit, report=report, sources=sources
        )

    if use_llm:
        items = estimate_with_llm(
            items,
            batch_size=batch_size,
            cache=cache if cache is not None else EstimateCache(enabled=use_cache),
            report=report,
            sources=sources,
            judge=judge,
            override_heat=override_heat,
        )

    enriched = [
        item.model_copy(update={"note": with_sources(item.note, sources[index])})
        if sources.get(index)
        else item
        for index, item in enumerate(items)
    ]
    return enriched, report
