"""表格导入工具：把 CSV / Excel 商品表导入选品库。

适用于走不了官方 API 的渠道 —— 1688 商家后台/分销后台导出的商品表、
第三方数据服务导出的 Excel、手工整理的 CSV。

典型流程：

    # 1. 先看自动识别成什么样，确认无误再入库（不写库）
    python scripts/import_table.py --file 1688导出.csv --preview

    # 2. 确认后入库，并把映射存成方案，下次自动复用
    python scripts/import_table.py --file 1688导出.csv --import \\
        --save-profile 1688 --markup 2.5 --score

    # 3. 再导入同一份报表时，映射会自动套用，无需重复配置
    python scripts/import_table.py --file 1688导出-新一期.csv --import --score

自动识别不准时用 --map 手工指定：

    python scripts/import_table.py --file 表.xlsx --preview \\
        --map "商品标题=产品名,成本=供货价,重量=单件重量(g)" --weight-unit g
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, service  # noqa: E402
from app.config import settings  # noqa: E402
from app.tabular import (  # noqa: E402
    FIELD_LABELS,
    PRICE_UNIT_LABELS,
    WEIGHT_UNIT_LABELS,
    ColumnMapping,
    parse_mapping,
)

#: argparse choices 用
WEIGHT_UNITS = tuple(sorted(WEIGHT_UNIT_LABELS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导入 CSV / Excel 商品表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file", required=True, help="CSV / TSV / xlsx 文件路径")
    parser.add_argument("--sheet", default=0, help="Excel 工作表名或序号（默认 0）")
    parser.add_argument("--encoding", default=None, help="强制指定 CSV 编码（默认自动识别）")
    parser.add_argument("--delimiter", default=None, help="强制指定分隔符（默认自动识别）")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="只预览，不写库（默认行为）")
    mode.add_argument("--import", dest="do_import", action="store_true", help="确认导入数据库")

    parser.add_argument("--map", default="", metavar="K=COL,...",
                        help='手工指定列映射，如 "商品标题=产品名,成本=供货价"')
    parser.add_argument("--profile", default=None, help="套用已保存的映射方案名")
    parser.add_argument("--save-profile", default="", metavar="NAME",
                        help="把本次使用的映射存为该名字的方案")
    parser.add_argument("--profiles", action="store_true", help="列出已保存的映射方案")
    parser.add_argument("--delete-profile", default=None, metavar="NAME", help="删除映射方案")

    parser.add_argument("--markup", type=float, default=2.5,
                        help="只有采购价时推导售价的加价倍数（默认 2.5）")
    parser.add_argument("--weight-unit", default=None, choices=WEIGHT_UNITS,
                        help="重量列单位（默认从列名识别）")
    parser.add_argument("--price-unit", default=None, choices=sorted(PRICE_UNIT_LABELS),
                        help="金额列单位（默认从列名识别）")
    parser.add_argument("--source-label", default="", help="来源标签，默认用文件名")

    parser.add_argument("--score", action="store_true", help="导入后立即打分")
    parser.add_argument("--use-llm", action="store_true", help="打分时附带大模型点评")
    parser.add_argument("--top", type=int, default=10, help="预览/打分后打印前 N 条")
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> dict:
    sheet: object = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    return {
        "sheet": sheet,
        "encoding": args.encoding,
        "delimiter": args.delimiter,
        "markup": args.markup,
        "weight_unit": args.weight_unit,
        "price_unit": args.price_unit,
        "source_label": args.source_label or None,
    }


def print_mapping(mapping: ColumnMapping, columns: list[str]) -> None:
    print("列映射：")
    print(f"  {'字段':<18}{'中文名':<12}{'源列'}")
    print("  " + "-" * 66)
    for target in FIELD_LABELS:
        column = mapping.fields.get(target)
        mark = "" if column else "  （未映射，取缺省值）"
        print(f"  {target:<20}{FIELD_LABELS[target]:<14}{column or '—'}{mark}")
    unused = [column for column in columns if column not in mapping.fields.values()]
    if unused:
        print(f"\n  表内未使用的列：{'、'.join(unused)}")
    print(f"\n  重量单位：{WEIGHT_UNIT_LABELS.get(mapping.weight_unit, mapping.weight_unit)}"
          f"　金额单位：{PRICE_UNIT_LABELS.get(mapping.price_unit, mapping.price_unit)}"
          f"　加价倍数：{mapping.markup:g}")


def cmd_profiles() -> int:
    profiles = db.list_mapping_profiles()
    if not profiles:
        print("还没有保存任何映射方案。用 --import --save-profile 名字 保存一个。")
        return 0
    print(f"{'ID':<5}{'名称':<22}{'列数':<6}{'指纹':<18}源列")
    print("-" * 110)
    for profile in profiles:
        columns = profile["columns"]
        preview = "、".join(columns[:4]) + ("…" if len(columns) > 4 else "")
        print(f"{profile['id']:<7}{profile['name'][:20]:<24}{len(columns):<8}"
              f"{profile['fingerprint']:<20}{preview}")
    return 0


def cmd_preview(args: argparse.Namespace, explicit: dict | None) -> int:
    try:
        data, mapping, result = service.preview_table(
            args.file, mapping=explicit, **build_options(args)
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"读表失败：{exc}", file=sys.stderr)
        return 2

    print(f"文件：{args.file}")
    print(f"表格：{data.describe()}\n")
    print_mapping(mapping, data.columns)

    print(f"\n解析结果：{result.summary()}")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")

    if result.products:
        print(f"\n{'#':<4}{'售价':<10}{'成本':<10}{'毛利率':<9}{'热度':<8}{'重量kg':<9}{'商品'}")
        print("-" * 92)
        for index, item in enumerate(result.products[: args.top], start=1):
            margin = (item.price - item.cost) / item.price if item.price else 0
            print(f"{index:<6}{item.price:<12.2f}{item.cost:<12.2f}{margin:<11.0%}"
                  f"{item.heat:<10.1f}{item.weight_kg:<11.2f}{item.title[:26]}")
        print(f"\n数据说明：{result.provenance}")

    print("\n（--preview 模式，未写入数据库。确认无误后加 --import）")
    return 0


def cmd_import(args: argparse.Namespace, explicit: dict | None) -> int:
    try:
        report = service.import_table(
            args.file,
            mapping=explicit,
            save_as=args.save_profile,
            **build_options(args),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"导入失败：{exc}", file=sys.stderr)
        return 2

    print(f"已导入 {report['saved']} 个商品 → {settings.db_path}")
    if report["skipped"]:
        print(f"跳过 {report['skipped']} 行")
    for warning in report["warnings"]:
        print(f"  ⚠️  {warning}")
    print(f"数据说明：{report['provenance']}")
    if args.save_profile:
        print(f"映射已存为方案「{args.save_profile}」，下次导入同类报表会自动套用。")

    if args.score:
        results = service.score_all(use_llm=args.use_llm)
        print(f"\n{'排名':<4}{'总分':<8}{'等级':<6}{'商品'}")
        print("-" * 64)
        for index, item in enumerate(results[: args.top], start=1):
            print(f"{index:<6}{item['total']:<10}{item['grade']:<8}{item['title'][:32]}")
        print(f"\n完成，共打分 {len(results)} 个商品。")
    return 0


def main() -> int:
    args = parse_args()
    service.prepare_db()

    if args.profiles:
        return cmd_profiles()
    if args.delete_profile:
        if db.delete_mapping_profile(args.delete_profile):
            print(f"已删除映射方案「{args.delete_profile}」。")
            return 0
        print(f"未找到映射方案「{args.delete_profile}」。", file=sys.stderr)
        return 1

    try:
        explicit = parse_mapping(args.map) or None
    except ValueError as exc:
        print(f"映射解析失败：{exc}", file=sys.stderr)
        return 2

    if explicit is None and args.profile:
        record = db.get_mapping_profile(args.profile)
        if record is None:
            print(f"未找到映射方案「{args.profile}」。用 --profiles 查看。", file=sys.stderr)
            return 1
        explicit = record["mapping"]

    return cmd_import(args, explicit) if args.do_import else cmd_preview(args, explicit)


if __name__ == "__main__":
    raise SystemExit(main())
