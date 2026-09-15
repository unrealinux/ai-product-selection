"""成本对齐：把 1688 / 表格导入的供货价配到淘宝商品的零售价上，算出真实毛利率。

为什么需要这一步
    淘宝 A2A 只给零售价、没有成本 → 所有商品毛利率 100% → 权重最高的维度静默饱和。
    1688 表格导入相反：有采购价、没有零售价。两边配起来毛利率才第一次是真实数据。

典型流程：

    # 1. 先导入 1688 商品表（提供采购价）
    python scripts/import_table.py --file 1688导出.csv --import --save-profile 1688

    # 2. 再拉淘宝商品（提供真实零售价）
    python scripts/taobao_fetch.py --queries "保温杯" --limit 20 --import

    # 3. 预览对齐结果 —— 默认不写库，因为标题匹配是启发式，需要人看一眼
    python scripts/link_costs.py --preview

    # 4. 确认后写入，再重新打分
    python scripts/link_costs.py --apply --score

⚠️ 匹配是启发式的
    跨平台没有共同商品 ID，只能靠标题相似度。高分≠同一款货，只代表"很可能是同类"。
    所以默认只预览，每条匹配都带相似度，低置信的单独列出、默认不应用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, service  # noqa: E402
from app.config import settings  # noqa: E402
from app.costlink import (  # noqa: E402
    DEFAULT_THRESHOLD,
    apply_matches,
    link_costs,
    margin_report,
    title_similarity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把供货价对齐到零售商品上，算出真实毛利率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", default=True,
                      help="只预览，不写库（默认行为）")
    mode.add_argument("--apply", action="store_true", help="确认写入数据库")

    parser.add_argument("--target-source", default="",
                        help="只补这个来源的商品（如 taobao）；留空=所有缺成本的商品")
    parser.add_argument("--supply-source", default="",
                        help="只从这个来源取成本（如 1688导出示例）；留空=所有有成本的商品")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"相似度阈值（默认 {DEFAULT_THRESHOLD}）")
    parser.add_argument("--min-specs", type=int, default=0,
                        help="要求至少共享几个规格 token（如 304/500ml），0=不要求")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有成本（默认不覆盖：已有成本更可信）")
    parser.add_argument("--include-low-confidence", action="store_true",
                        help="连同低置信候选一起写入（不建议）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个待补商品")
    parser.add_argument("--top", type=int, default=15, help="打印前 N 条匹配")

    parser.add_argument("--explain", default="", metavar="标题A|标题B",
                        help="对两个标题算相似度，用于校准阈值")
    parser.add_argument("--score", action="store_true", help="写入后重新打分")
    return parser.parse_args()


def cmd_explain(raw: str) -> int:
    if "|" not in raw:
        print("格式：--explain \"标题A|标题B\"", file=sys.stderr)
        return 2
    left, _, right = raw.partition("|")
    print(f"A: {left.strip()}")
    print(f"B: {right.strip()}")
    print(f"相似度：{title_similarity(left.strip(), right.strip()):.4f}")
    return 0


def print_matches(result, top: int) -> None:
    accepted = sorted(result.accepted, key=lambda m: -m.score)
    if accepted:
        print(f"\n高置信匹配（相似度 ≥ 阈值，{len(accepted)} 条）：")
        print(f"  {'相似度':<9}{'售价':<10}{'成本':<10}{'毛利率':<9}{'共同规格':<14}"
              f"{'淘宝商品 ← 供货来源'}")
        print("  " + "-" * 108)
        for match in accepted[:top]:
            specs = "/".join(match.shared_specs)[:12]
            print(f"  {match.score:<11.3f}{match.target.price:<12.2f}{match.cost:<12.2f}"
                  f"{match.margin:<11.1%}{specs:<16}"
                  f"{match.target.title[:26]} ← {match.supply_title[:26]}")

    low = sorted(result.low_confidence, key=lambda m: -m.score)
    if low:
        print(f"\n低置信候选（未达阈值，默认不应用，{len(low)} 条）：")
        for match in low[:top]:
            print(f"  {match.score:.3f}  {match.target.title[:26]} ← {match.supply_title[:26]}")

    unmatched = [m for m in result.matches if m.supply is None]
    if unmatched:
        print(f"\n无匹配（{len(unmatched)} 条，前 {min(top, len(unmatched))} 条）：")
        for match in unmatched[:top]:
            print(f"  {match.target.title[:60]}")


def main() -> int:
    args = parse_args()
    if args.explain:
        return cmd_explain(args.explain)

    service.prepare_db()
    products = db.list_products(limit=5000)
    if not products:
        print("库里没有商品。先导入数据。", file=sys.stderr)
        return 1

    print("对齐前：")
    before = margin_report(products)
    print(f"  商品 {len(products)} 个｜有成本 {before['count']} 个｜"
          f"缺成本 {before['missing_cost']} 个｜毛利率≥95%（等于没成本）{before['saturated']} 个")
    if before["count"]:
        print(f"  已有成本的毛利率：中位数 {before['median']:.1%}")

    result = link_costs(
        products,
        target_source=args.target_source,
        supply_source=args.supply_source,
        threshold=args.threshold,
        overwrite=args.overwrite,
        min_shared_specs=args.min_specs,
        dry_run=not args.apply,
        limit=args.limit,
    )

    # 已有成本且不允许覆盖的，单独报一下
    if not args.overwrite:
        skipped = [p for p in products if p.cost > 0
                   and (not args.target_source or p.source == args.target_source)]
        result.skipped_existing = len(skipped)

    print(f"\n{result.summary()}")
    for warning in result.warnings:
        print(f"  ⚠️  {warning}")

    if result.matches:
        print_matches(result, args.top)

    if any(m.supply is not None for m in result.matches) and not args.apply:
        print("\n（--preview 模式，未写入数据库。确认匹配质量后加 --apply）")
        print("  调阈值：--threshold 0.55　校准阈值：--explain \"标题A|标题B\"")
        return 0

    if args.apply:
        written = apply_matches(result, include_low_confidence=args.include_low_confidence)
        print(f"\n已写入 {written} 条成本对齐结果 → {settings.db_path}")
        if not args.include_low_confidence and result.low_confidence:
            print(f"（{len(result.low_confidence)} 条低置信候选未写入；"
                  f"确认无误可加 --include-low-confidence）")

        after = margin_report(db.list_products(limit=5000))
        print(f"\n对齐后：有成本 {after['count']} 个｜缺成本 {after['missing_cost']} 个｜"
              f"毛利率≥95% {after['saturated']} 个")
        if after["count"]:
            print(f"  毛利率：中位数 {after['median']:.1%}，"
                  f"区间 {after['min']:.1%} ~ {after['max']:.1%}")

        if args.score:
            results = service.score_all()
            print(f"\n{'排名':<4}{'总分':<8}{'等级':<6}{'毛利率':<9}{'商品'}")
            print("-" * 76)
            for index, item in enumerate(results[: args.top], start=1):
                print(f"{index:<6}{item['total']:<10}{item['grade']:<8}"
                      f"{item['profit_margin']:<11.0%}{item['title'][:34]}")
            print(f"\n完成，共打分 {len(results)} 个商品。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
