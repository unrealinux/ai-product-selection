"""录入真实经营结果，供「哪套权重更赚钱」回测使用。

典型用法：

    # 1. 先看表里的列被识别成了什么，以及每行靠什么键匹配上商品（不写库）
    python scripts/record_outcome.py --file 生意参谋导出.csv --preview

    # 2. 确认后写入。同一 (商品, 起止日期) 重复导入会更新而非新增
    python scripts/record_outcome.py --file 生意参谋导出.csv --import --source 生意参谋

    # 3. 表里没有日期列时，用参数补；单日导出的「日期」列会被当作起止同一天
    python scripts/record_outcome.py --file 结果.csv --import --start 2024-03-01 --end 2024-03-31

商品匹配顺序（每行都会告诉你用的是哪一个）：

    平台商品ID → 表内「商品ID」（先当库内主键，再当平台 ID） → 标题精确 → 标题模糊

真实导出表的「商品ID」是**平台的 ID**，本库把它存在 ``external_id`` 里，所以两条路都会试。
标题对不上时按二元组相似度模糊匹配（阈值 ``--title-threshold``，默认 0.75），
歧义候选会一并列出。匹配不上的行会逐条列出，**不会静默写入 0**。

数据来源说明：结果表的每一行都记录了 ``source``，回测结论里会标明数据来自哪里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, service  # noqa: E402
from app.config import settings  # noqa: E402
from app.outcome_import import (  # noqa: E402
    DEFAULT_TITLE_THRESHOLD,
    NUMERIC_FIELDS,
    ProductIndex,
    build_column_map,
)
from app.outcomes import parse_day  # noqa: E402
from app.tabular import parse_number, read_table  # noqa: E402


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
    parser.add_argument("--title-threshold", type=float, default=DEFAULT_TITLE_THRESHOLD,
                        help=f"标题模糊匹配阈值，默认 {DEFAULT_TITLE_THRESHOLD}")
    parser.add_argument("--top", type=int, default=20, help="预览时最多打印几行")
    return parser.parse_args()


def _collect(rows: list[dict], column_map: dict[str, str]) -> list[dict]:
    """把表格行转成 ``{目标字段: 值}``。同一目标有多列时保留第一个。"""
    collected: list[dict] = []
    for row in rows:
        values: dict[str, object] = {}
        for column, target in column_map.items():
            if target in values:
                continue
            values[target] = row.get(column)
        collected.append(values)
    return collected


def main() -> int:
    args = parse_args()
    service.prepare_db()

    kwargs = {"sheet": args.sheet} if args.sheet else {}
    table = read_table(args.file, **kwargs)
    print(f"读取 {args.file}：{table.describe()}")

    column_map = build_column_map(table.columns)
    mapped = set(column_map.values())
    print("\n列映射：")
    for column, target in column_map.items():
        print(f"  {column}  →  {target}")
    unmapped = [c for c in table.columns if c not in column_map]
    if unmapped:
        print(f"  未识别（忽略）：{', '.join(unmapped)}")

    if not ({"product_id", "external_id", "title"} & mapped):
        print("\n❌ 表里没有商品 ID、外部编号或标题列，无法匹配商品。")
        return 1

    start_default = parse_day(args.start) if args.start else None
    end_default = parse_day(args.end) if args.end else None
    if args.start and not start_default:
        print(f"❌ --start 不是合法日期：{args.start!r}")
        return 1
    if args.end and not end_default:
        print(f"❌ --end 不是合法日期：{args.end!r}")
        return 1

    products = db.list_products(limit=10000)
    if not products:
        print("❌ 库里没有商品。先导入商品再录结果。")
        return 1
    index = ProductIndex(products, title_threshold=args.title_threshold)

    records: list[dict] = []
    skipped: list[str] = []
    ambiguous: list[str] = []

    for values in _collect(table.rows, column_map):
        row_no = len(records) + len(skipped) + 2  # 表头占第 1 行

        match = index.match(
            raw_id=values.get("product_id"),
            external_id=values.get("external_id"),
            title=values.get("title"),
        )
        if not match.matched:
            label = values.get("product_id") or values.get("external_id") or values.get("title") or "?"
            skipped.append(f"第 {row_no} 行：匹配不到商品（{label}）")
            continue
        if match.ambiguous:
            best_alt = match.alternatives[0]
            ambiguous.append(
                f"第 {row_no} 行：『{str(values.get('title') or '')[:20]}』"
                f" ≈ {match.product.title[:20]}（{match.score:.3f}）"
                f"，但 {best_alt[0].title[:20]} 也有 {best_alt[1]:.3f}"
            )

        date_value = parse_day(values.get("window_date")) if values.get("window_date") else None
        start = date_value or parse_day(values.get("window_start")) or start_default
        end = date_value or parse_day(values.get("window_end")) or end_default
        if not start or not end:
            skipped.append(f"第 {row_no} 行：缺开始/结束日期（可用 --start --end 补充）")
            continue

        payload = {field: (parse_number(values.get(field)) or 0.0) for field in NUMERIC_FIELDS}
        negative = [field for field in NUMERIC_FIELDS if payload[field] < 0]
        if negative:
            skipped.append(f"第 {row_no} 行：{'、'.join(negative)} 为负数，需先在表里改正")
            continue

        note = str(values.get("note") or "").strip()
        source = (args.source or str(values.get("source") or "").strip()
                  or f"表格导入（{Path(args.file).name}）")
        records.append({
            "product_id": match.product.id,
            "window_start": start,
            "window_end": end,
            "title": match.product.title,
            "match_key": match.describe(),
            "match_score": match.score,
            **payload,
            "note": note,
            "source": source,
        })

    if not records:
        print("\n没有可写入的行。")
        for reason in skipped[: args.top]:
            print(f"  ⚠️  {reason}")
        return 1

    by_key: dict[str, int] = {}
    for record in records:
        by_key[record["match_key"]] = by_key.get(record["match_key"], 0) + 1
    print(f"\n匹配方式：{'；'.join(f'{key} {count} 行' for key, count in sorted(by_key.items()))}")
    print(f"可写入 {len(records)} 行（跳过 {len(skipped)} 行）")

    print(f"\n{'商品':<28}{'窗口':<24}{'曝光':>8}{'点击':>8}{'订单':>7}{'成交额':>11}  {'匹配':<12}")
    print("-" * 108)
    for item in records[: args.top]:
        window = f"{item['window_start']} ~ {item['window_end']}"
        print(f"{item['title'][:26]:<28}{window:<24}"
              f"{item['impressions']:>8.0f}{item['clicks']:>8.0f}"
              f"{item['orders']:>7.0f}{item['revenue']:>11.1f}  {item['match_key']:<12}")
    if len(records) > args.top:
        print(f"  …… 其余 {len(records) - args.top} 行省略（--top 调整）")

    if ambiguous:
        print(f"\n⚠️  {len(ambiguous)} 行模糊匹配存在歧义，建议核对（低置信可调 --title-threshold）：")
        for reason in ambiguous[: args.top]:
            print(f"  {reason}")
    for reason in skipped[: args.top]:
        print(f"  ⚠️  {reason}")

    if not args.do_import:
        print("\n（预览模式，未写入。确认无误后加 --import）")
        return 0

    written = 0
    failed: list[str] = []
    for item in records:
        payload = {key: value for key, value in item.items()
                   if key not in {"product_id", "window_start", "window_end",
                                  "title", "match_key", "match_score"}}
        try:
            service.record_outcome(item["product_id"], item["window_start"], item["window_end"],
                                   **payload)
        except (ValueError, KeyError) as exc:
            failed.append(f"{item['title'][:20]}：{exc}")
            continue
        written += 1

    print(f"\n✅ 已写入 {written} 条经营结果 → {settings.db_path}")
    for reason in failed[: args.top]:
        print(f"  ❌ {reason}")
    print("   下一步：python scripts/backtest.py --doctor  看数据是否够回测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
