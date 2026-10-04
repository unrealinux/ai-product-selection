"""把真实经营结果表映射成可回测的记录。

真实导出表为什么不能直接进回测
------------------------------
电商后台导出的表（生意参谋 / 抖店 / 淘宝订单）有两个和本项目对不上的地方：

1. **表里的「商品ID」是平台的 ID，不是本库的主键**。本库把平台 ID 存在
   ``external_id`` 里，所以必须能按平台 ID 匹配 —— 否则整批行都匹配不上。
2. **标题对不上**。平台导出的标题常带促销后缀、括号、不同分隔符，
   精确匹配会大面积失败，而失败的行会被丢掉。

这里的匹配顺序是：明确的外部编号 → 表内「商品ID」（先当内部主键，再当平台 ID）
→ 标题精确 → **标题模糊**（复用成本对齐那套二元组相似度）。
每一行都会返回它**是靠哪个键匹配上的**，以及模糊匹配的相似度与其它候选，
方便人工复核。

设计原则：宁可报告「匹配不上」，也不猜。模糊匹配有阈值，且歧义候选会一并列出。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .costlink import bigrams, title_similarity
from .models import ProductIn
from .tabular import normalize_column

#: 目标字段 → 可能出现的列名（中英文别名）。真实导出表里这几种写法都常见
ALIASES: dict[str, tuple[str, ...]] = {
    "product_id": ("商品id", "商品编号", "宝贝id", "product_id", "productid", "id"),
    "external_id": (
        "平台商品id", "外部编号", "商家编码", "货号", "商品编码",
        "offerid", "offer_id", "itemid", "item_id",
        "淘宝商品id", "抖音商品id", "商品链接id",
    ),
    "title": ("商品标题", "商品名称", "宝贝标题", "标题", "产品名称",
              "title", "product_name", "item_name"),
    "window_start": ("开始日期", "起始日期", "统计开始", "开始时间", "起始",
                     "window_start", "start", "date_start"),
    "window_end": ("结束日期", "截止日期", "统计结束", "结束时间", "截止",
                   "window_end", "end", "date_end"),
    # 单日导出：一列「日期」，此时 start = end = 该日期
    "window_date": ("日期", "统计日期", "数据日期", "date", "dt", "day"),
    "impressions": ("曝光", "展现", "曝光量", "展现量", "曝光次数", "展现次数",
                    "商品曝光", "impressions", "impression"),
    "clicks": ("点击", "点击量", "点击次数", "商品点击", "clicks", "click"),
    "orders": ("订单", "订单数", "订单量", "支付订单", "支付子订单数", "成交订单数",
               "orders", "order_count"),
    "units": ("销量", "件数", "支付件数", "支付商品件数", "成交件数",
              "units", "quantity", "qty"),
    "returns": ("退货", "退货件数", "退货量", "退货数", "退款", "退款笔数",
                "成功退款笔数", "returns", "refund_count"),
    "revenue": ("成交金额", "成交额", "销售额", "支付金额", "gmv",
                "revenue", "sales_amount", "amount"),
    "cogs": ("成本", "采购成本", "采购金额", "成本金额", "商品成本", "cogs"),
    "ad_spend": ("推广花费", "推广费用", "推广消费", "广告花费", "花费",
                 "ad_spend", "ad_cost", "adcost"),
    "note": ("备注", "说明", "remark", "note", "comment"),
    "source": ("来源", "渠道", "source", "channel"),
}

#: 结果表里的数值字段（商品与日期之外）
NUMERIC_FIELDS: tuple[str, ...] = (
    "impressions", "clicks", "orders", "units", "returns", "revenue", "cogs", "ad_spend",
)

#: 模糊匹配时最多收集多少候选商品
CANDIDATE_LIMIT = 200

#: 默认模糊匹配阈值。低于它视为匹配不上
DEFAULT_TITLE_THRESHOLD = 0.75

#: 与最佳候选相差在这个范围内，算歧义，会一并列出来
AMBIGUITY_MARGIN = 0.05


def build_column_map(columns: Sequence[str]) -> dict[str, str]:
    """列名 → 目标字段。两轮匹配：精确（归一化后）→ 包含。"""
    normalized = {column: normalize_column(column) for column in columns}
    alias_norm = {
        target: [normalize_column(alias) for alias in aliases]
        for target, aliases in ALIASES.items()
    }
    mapping: dict[str, str] = {}
    for target, aliases in alias_norm.items():
        for column, key in normalized.items():
            if key in aliases:
                mapping[column] = target
                break
    for target, aliases in alias_norm.items():
        if target in mapping.values():
            continue
        for column, key in normalized.items():
            if column in mapping:
                continue
            if any(alias and alias in key for alias in aliases):
                mapping[column] = target
                break
    return mapping


# --------------------------------------------------------------------------- #
# 商品匹配
# --------------------------------------------------------------------------- #

@dataclass
class ProductMatch:
    """一行结果对应的商品，以及它是靠哪个键匹配上的。"""

    product: Optional[ProductIn] = None
    key: str = "none"
    score: float = 1.0
    #: 其它相近候选（模糊匹配时），``(商品, 相似度)``
    alternatives: list[tuple[ProductIn, float]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.product is not None

    @property
    def ambiguous(self) -> bool:
        """最佳候选与次优候选太接近 —— 需要人工确认。"""
        if not self.alternatives:
            return False
        return (self.score - self.alternatives[0][1]) <= AMBIGUITY_MARGIN

    def describe(self) -> str:
        if not self.matched:
            return "匹配不上"
        labels = {
            "product_id": "库内主键",
            "external_id": "平台商品ID",
            "title": "标题精确",
            "title_fuzzy": f"标题模糊 {self.score:.3f}",
        }
        text = labels.get(self.key, self.key)
        return text + ("（有歧义）" if self.ambiguous else "")


class ProductIndex:
    """商品匹配索引。一次构建，多行复用。"""

    def __init__(self, products: Iterable[ProductIn], *,
                 title_threshold: float = DEFAULT_TITLE_THRESHOLD,
                 candidate_limit: int = CANDIDATE_LIMIT) -> None:
        self.products = list(products)
        self.title_threshold = title_threshold
        self.candidate_limit = candidate_limit

        self._by_id: dict[int, ProductIn] = {}
        self._by_external: dict[str, ProductIn] = {}
        self._by_title: dict[str, ProductIn] = {}
        self._bigram_index: dict[str, list[ProductIn]] = defaultdict(list)

        for product in self.products:
            self._by_id[product.id] = product
            if product.external_id:
                self._by_external.setdefault(product.external_id.strip(), product)
            self._by_title.setdefault(product.title.strip(), product)
            for gram in bigrams(product.title):
                self._bigram_index[gram].append(product)

    def match(self, *, raw_id: Any = None, external_id: Any = None,
              title: Any = None) -> ProductMatch:
        """按优先级匹配一行结果。

        ``raw_id`` 是表里那一列「商品ID」—— 它可能是本库主键（自己导出的模板），
        也可能是平台 ID（生意参谋 / 抖店导出）。两种都试。
        """
        explicit = str(external_id).strip() if external_id not in (None, "") else ""
        if explicit:
            hit = self._by_external.get(explicit)
            if hit is not None:
                return ProductMatch(hit, "external_id")

        if raw_id not in (None, ""):
            text = str(raw_id).strip()
            digits = text.replace(",", "").replace("，", "")
            # 只有整串是数字才当库内主键 —— “A002” 这种货号里也含数字，
            # 用宽松的数字提取会把它误当成主键 2
            if digits.isdigit():
                hit = self._by_id.get(int(digits))
                if hit is not None:
                    return ProductMatch(hit, "product_id")
            hit = self._by_external.get(text)
            if hit is not None:
                return ProductMatch(hit, "external_id")

        text = str(title).strip() if title not in (None, "") else ""
        if text:
            hit = self._by_title.get(text)
            if hit is not None:
                return ProductMatch(hit, "title")
            return self._fuzzy(text)
        return ProductMatch(None, "none")

    def _candidates(self, row_bigrams: set[str]) -> list[ProductIn]:
        """用倒排索引收候选：从「更稀有的 bigram」开始，避免常用字带来巨量候选。"""
        ordered = sorted(row_bigrams, key=lambda gram: len(self._bigram_index.get(gram, ())))
        found: dict[int, ProductIn] = {}
        for gram in ordered:
            for product in self._bigram_index.get(gram, ()):
                found[product.id] = product
                if len(found) >= self.candidate_limit:
                    return list(found.values())
        return list(found.values())

    def _fuzzy(self, title: str) -> ProductMatch:
        row_bigrams = bigrams(title)
        if not row_bigrams:
            return ProductMatch(None, "none")

        scored: list[tuple[ProductIn, float]] = []
        for product in self._candidates(row_bigrams):
            score = title_similarity(title, product.title)
            if score >= self.title_threshold:
                scored.append((product, score))
        if not scored:
            return ProductMatch(None, "none")

        scored.sort(key=lambda item: item[1], reverse=True)
        best, score = scored[0]
        return ProductMatch(best, "title_fuzzy", score=score, alternatives=scored[1:5])
