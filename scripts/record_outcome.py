"""录入真实经营结果，供「哪套权重更赚钱」回测使用。

典型用法：

    # 1. 先看表里的列被识别成了什么（不写库）
    python scripts/record_outcome.py --file 生意参谋导出.csv --preview

    # 2. 确认后写入。同一 (商品, 起止日期) 重复导入会更新而非新增
    python scripts/record_outcome.py --file 生意参谋导出.csv --import --source 生意参谋

    # 3. 表里没有日期列时，用参数补
    python scripts/record_outcome.py --file 结果.csv --import --start 2024-03-01 --end 2024-03-31

商品按表里的商品 ID 匹配；没有 ID 列时退回标题精确匹配。匹配不上的行会被
逐条列出来，**不会静默写入 0**。

数据来源说明：结果表的每一行都记录了 ``source``，回测结论里会标明数据来自哪里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, service  # noqa: E402
from app.config import settings  # noqa: E402
from app.outcomes import parse_day  # noqa: E402
from app.tabular import normalize_column, parse_number, read_table  # noqa: E402

#: 目标字段 → 可能出现的列名（中英文别名）
ALIASES: dict[str, tuple[str, ...]] = {
    "product_id": ("商品id", "商品ID", "product_id", "productid", "宝贝id", "id"),
    "title": ("商品标题", "标题", "商品名称", "title", "product_name"),
    "window_start": ("开始日期", "起始日期", "window_start", "start", "起始"),
    "window_end": ("结束日期", "截止日期", "window_end", "end", "截止"),
    "impressions": ("曝光", "展现", "曝光量", "impressions"),
    "clicks": ("点击", "点击量", "clicks"),
    "orders": ("订单", "订单数", "支付订单", "orders"),
    "units": ("销量", "件数", "支付件数", "units"),
    "returns": ("退货", "退货件数", "退款", "returns"),
    "revenue": ("成交金额", "销售额", "支付金额", "gmv", "revenue"),
    "cogs": ("成本", "采购成本", "商品成本", "cogs"),
    "ad_spend": ("推广花费", "广告花费", "花费", "ad_spend"),
    "note": ("备注", "note"),
    "source": ("来源", "source"),
}

#: 结果表里除商品与日期外的数值字段
NUMERIC_FIELDS: tuple[str, ...] = (
    "impressions", "clicks", "orders", "units", "returns", "revenue", "cogs", "ad_spend",
)


def build_column_map(columns: list[str]) -> dict[str, str]:
    """列名 → 目标字段。两轮匹配：精确（归一化后）→ 包含。"""
    normalized = {column: normalize_column(column) for column in columns}
    alias_norm = {
        field: [normalize_column(alias) for alias in aliases]
        for field, aliases in ALIASES.items()
    }
    mapping: dict[str, str] = {}
    for field, aliases in alias_norm.items():
        for column, key in normalized.items():
            if key in aliases:
                mapping[column] = field
                break
    for field, aliases in alias_norm.items():
        if field in mapping.values():
            continue
        for column, key in normalized.items():
            if column in mapping:
                continue
            if any(alias and alias in key for alias in aliases):
                mapping[column] = field
                break
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="录入真实经营结果（用于效果回测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file", required=True, help="CSV / TSV / Excel 文件")
    parser.add_argument("--sheet", default=None, help="Excel 工作表名（默认第一个）")
    parser.add_argument("--import", dest="do_import", action="store_true",
                        help="真正写入数据库（默认只预览）")
    parser.add_argument("--preview", action="store_true",
                        help="只预览不写库（默认行为，显式写上更清楚）")
    parser.add_argument("--start", default=None, help="表里没有开始日期列时使用")
    parser.add_argument("--end", default=None, help="表里没有结束日期列时使用")
    parser.add_argument("--source", default="", help="覆盖来源标签")
    parser.add_argument("--match-by", choices=["auto", "title", "product_id"], default="auto",
                        help="商品匹配方式，默认自动")
    parser.add_argument("--top", type=int, default=20, help="预览时最多打印几行")
    return parser.parse_args()


def resolve_products(match_by: str) -> dict[str, object]:
    """构造 标题 → 商品的索引，用于没有 ID 列时的匹配。"""
    if match_by == "product_id":
        return {}
    return {item.title.strip(): item for item in db.list_products(limit=10000)}


def main() -> int:
    args = parse_args()
    service.prepare_db()

    kwargs = {"sheet": args.sheet} if args.sheet else {}
    table = read_table(args.file, **kwargs)
    print(f"读取 {args.file}：{table.describe()}")

    column_map = build_column_map(table.columns)
    mapped = set(column_map.values())
    print("\n列映射：")
    for column, field in column_map.items():
        print(f"  {column}  →  {field}")
    unmapped = [c for c in table.columns if c not in column_map]
    if unmapped:
        print(f"  未识别（忽略）：{', '.join(unmapped)}")

    if "product_id" not in mapped and "title" not in mapped:
        print("\n❌ 表里既没有商品 ID 也没有标题列，无法匹配商品。")
        print("   用 --file 换成带「商品ID」或「商品标题」的表。")
        return 1

    start_default = parse_day(args.start) if args.start else None
    end_default = parse_day(args.end) if args.end else None
    if args.start and not start_default:
        print(f"❌ --start 不是合法日期：{args.start!r}")
        return 1
    if args.end and not end_default:
        print(f"❌ --end 不是合法日期：{args.end!r}")
        return 1

    by_title = resolve_products(args.match_by)
    records: list[dict] = []
    skipped: list[str] = []

    for index, row in enumerate(table.rows, start=2):  # 表头占第 1 行
        values = {field: row.get(column) for column, field in column_map.items()}

        product = None
        if args.match_by != "title" and values.get("product_id") is not None:
            raw_id = parse_number(values.get("product_id"))
            if raw_id is not None:
                product = db.get_product(int(raw_id))
        if product is None and args.match_by != "product_id":
            title = str(values.get("title") or "").strip()
            if title:
                product = by_title.get(title)
        if product is None:
            label = values.get("product_id") or values.get("title") or "?"
            skipped.append(f"第 {index} 行：匹配不到商品（{label}）")
            continue

        start = parse_day(values.get("window_start")) or start_default
        end = parse_day(values.get("window_end")) or end_default
        if not start or not end:
            skipped.append(f"第 {index} 行：缺开始/结束日期（可用 --start --end 补充）")
            continue

        payload = {field: (parse_number(values.get(field)) or 0.0) for field in NUMERIC_FIELDS}
        negative = [field for field in NUMERIC_FIELDS if payload[field] < 0]
        if negative:
            # 负的曝光/订单一定是脏数据；负的成交额通常是退款没单列，宁可不写
            skipped.append(f"第 {index} 行：{'、'.join(negative)} 为负数，需先在表里改正")
            continue
        note = str(values.get("note") or "").strip()
        source = args.source or str(values.get("source") or "").strip() or f"表格导入（{Path(args.file).name}）"
        records.append({
            "product_id": product.id,
            "window_start": start,
            "window_end": end,
            "title": product.title,
            **payload,
            "note": note,
            "source": source,
        })

    if not records:
        print("\n没有可写入的行。")
        for reason in skipped[: args.top]:
            print(f"  ⚠️  {reason}")
        return 1

    records.sort(key=lambda item: item["product_id"])
    print(f"\n可写入 {len(records)} 行（跳过 {len(skipped)} 行）")
    print(f"\n{'商品':<28}{'窗口':<24}{'曝光':>8}{'点击':>8}{'订单':>7}{'成交额':>11}")
    print("-" * 90)
    for item in records[: args.top]:
        window = f"{item['window_start']} ~ {item['window_end']}"
        print(f"{item['title'][:26]:<28}{window:<24}"
              f"{item['impressions']:>8.0f}{item['clicks']:>8.0f}"
              f"{item['orders']:>7.0f}{item['revenue']:>11.1f}")
    if len(records) > args.top:
        print(f"  …… 其余 {len(records) - args.top} 行省略（--top 调整）")

    for reason in skipped[: args.top]:
        print(f"  ⚠️  {reason}")

    if not args.do_import:
        print("\n（预览模式，未写入。确认无误后加 --import）")
        return 0

    written = 0
    for item in records:
        payload = {key: value for key, value in item.items()
                   if key not in {"product_id", "window_start", "window_end", "title"}}
        service.record_outcome(item["product_id"], item["window_start"], item["window_end"],
                               **payload)
        written += 1

    print(f"\n✅ 已写入 {written} 条经营结果 → {settings.db_path}")
    print("   下一步：python scripts/backtest.py --runs 选一个快照回测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
