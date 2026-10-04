"""效果回测：用真实经营结果检验「哪套权重更赚钱」。

    # 有哪些指标可回测
    python scripts/backtest.py --metrics

    # 体检：结果数据够不够、有没有明显问题
    python scripts/backtest.py --doctor

    # 看有哪些快照可回测
    python scripts/backtest.py --runs

    # 回测单个快照（默认指标：毛利额）
    python scripts/backtest.py --run 1 --metric gross_profit

    # 只采用快照创建之后的结果窗口（推荐：打分之后的结果才构成预测）
    python scripts/backtest.py --run 1 --metric gross_profit --after-run-only

    # 对比两套权重谁更能预测结果
    python scripts/backtest.py --compare 1 2 --metric orders

    # 打印一张空的结果表模板，填完用 record_outcome.py 导入
    python scripts/backtest.py --template > 结果表.csv

重要：这是**相关性**，不是因果。被推的商品本身可能就更好卖。结论只能用来
筛掉「明显帮倒忙」的权重，不能当成「提升了 x% 利润」的证据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import service  # noqa: E402
from app.outcomes import (  # noqa: E402
    DEFAULT_METRIC,
    METRICS,
    MIN_SAMPLE,
    format_metric,
    metric_label,
)

TEMPLATE_HEADER = (
    "商品ID,商品标题,开始日期,结束日期,曝光,点击,订单数,销量,退货件数,"
    "成交金额,采购成本,推广花费,备注"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="效果回测：打分排序 vs 真实经营结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--metrics", action="store_true", help="列出可用指标")
    parser.add_argument("--runs", action="store_true", help="列出所有打分快照")
    parser.add_argument("--run", type=int, metavar="RUN_ID", help="回测指定快照")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("RUN_A", "RUN_B"),
                        help="对比两次快照的预测力")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help=f"回测指标，默认 {DEFAULT_METRIC}")
    parser.add_argument("--top-ratio", type=float, default=0.3,
                        help="Top 组占比，默认 0.3")
    parser.add_argument("--after-run-only", action="store_true",
                        help="只采用快照创建之后的结果窗口（推荐：打分之后的结果才构成预测）")
    parser.add_argument("--doctor", action="store_true",
                        help="体检：检查结果数据够不够回测、有没有明显问题")
    parser.add_argument("--top", type=int, default=20, help="最多打印几行，默认 20")
    parser.add_argument("--template", action="store_true", help="打印结果表 CSV 模板")
    return parser.parse_args()


def cmd_metrics() -> int:
    print("可用指标（higher_better=否 表示越小越好）：\n")
    print(f"  {'名称':<14}{'中文':<10}{'越大越好'}")
    print("-" * 40)
    for name, spec in METRICS.items():
        flag = "是" if spec["higher_better"] else "否（越低越好）"
        print(f"  {name:<14}{spec['label']:<10}{flag}")
    return 0


def cmd_runs() -> int:
    runs = service.list_snapshots(limit=100)
    if not runs:
        print("还没有打分快照。先建一个：python scripts/tune_weights.py --run \"基线\"")
        return 1
    print(f"{'ID':<6}{'商品数':<8}{'均分':<9}{'名称'}")
    print("-" * 60)
    for run in runs:
        print(f"{run['id']:<6}{run['product_count']:<8}{run['avg_score']:<9.2f}{run['label']}")
    print("\n回测：python scripts/backtest.py --run <ID> --metric gross_profit")
    return 0


def print_result(result, top: int) -> None:
    if not result.rows:
        print(f"\n{result.summary()}")
        for note in result.notes:
            print(f"  ⚠️  {note}")
        return

    metric = result.metric
    print(f"\n{'排名':<6}{'总分':<9}{metric_label(metric):<14}{'商品'}")
    print("-" * 78)
    ordered = sorted(result.rows, key=lambda row: row.rank)
    for row in ordered[:top]:
        print(f"{row.rank:<6}{row.score:<9.2f}"
              f"{format_metric(metric, row.value(metric)):<16}{row.title[:30]}")
    if len(ordered) > top:
        print(f"  …… 其余 {len(ordered) - top} 个省略（--top 调整）")

    print(f"\n{result.summary()}")
    if result.verdict:
        print(f"结论：{result.verdict}")
    for note in result.notes:
        print(f"  ⚠️  {note}")


def cmd_doctor() -> int:
    """回测前的体检：数据够不够、有没有明显问题。"""
    from collections import Counter

    from app import db
    from app.outcomes import MIN_SAMPLE, parse_day

    products = db.list_products(limit=100000)
    outcomes = db.list_outcomes(limit=100000)
    runs = service.list_snapshots(limit=100)

    print("=== 商品与结果 ===")
    with_outcome = {row["product_id"] for row in outcomes}
    if products:
        print(f"  商品 {len(products)} 个｜有结果 {len(with_outcome)} 个"
              f"（覆盖 {len(with_outcome) / len(products):.0%}）")
    else:
        print("  库里没有商品。")
    if outcomes:
        days = sorted(day for row in outcomes if (day := parse_day(row["window_start"])))
        per_product = Counter(row["product_id"] for row in outcomes)
        span = f"｜时间跨度 {days[0]} ~ {days[-1]}" if days else ""
        print(f"  结果记录 {len(outcomes)} 条{span}")
        print(f"  每个商品平均 {len(outcomes) / max(len(with_outcome), 1):.1f} 段"
              f"（最多 {max(per_product.values())} 段）")

    problems: list[str] = []
    if not outcomes:
        problems.append("还没有任何经营结果。先跑 scripts/record_outcome.py 导一版。")
    else:
        zero_rows = [row for row in outcomes
                     if not any(float(row.get(f) or 0) for f in
                                ("impressions", "clicks", "orders", "revenue"))]
        if zero_rows:
            problems.append(f"{len(zero_rows)} 条结果全部为 0 —— 可能是导错了列或抓错了行。")

        no_cost = [row for row in outcomes
                   if float(row.get("revenue") or 0) > 0 and float(row.get("cogs") or 0) <= 0]
        if no_cost:
            problems.append(
                f"{len(no_cost)} 条有成交额但没有采购成本 —— 算出的「毛利额」会偏高，"
                "建议把成本列也导进来。"
            )
        missing_cost_products = [p for p in products if p.cost <= 0]
        if missing_cost_products:
            problems.append(
                f"{len(missing_cost_products)} 个商品没有成本（未做成本对齐）——"
                "毛利类指标不可信，先跑 scripts/link_costs.py。"
            )
        if len(with_outcome) < MIN_SAMPLE:
            problems.append(
                f"只有 {len(with_outcome)} 个商品有结果，少于 {MIN_SAMPLE} 个时回测只会提示「样本不足」。"
            )

    print("\n=== 打分快照 ===")
    if not runs:
        print("  还没有快照。先建一个：python scripts/tune_weights.py --run \"基线\"")
        problems.append("还没有打分快照，回测没有可比对象。")
    else:
        all_ids = {row["product_id"] for row in outcomes}
        for run in runs:
            items = db.get_run_items(run["id"])
            comparable = sum(1 for item in items if item["product_id"] in all_ids)
            flag = "✅" if comparable >= MIN_SAMPLE else "⚠️"
            print(f"  {flag} #{run['id']} {run['label'][:24]:<26} 商品 {run['product_count']:<4}"
                  f"可比 {comparable:<4} 创建于 {run['created_at']:%Y-%m-%d}")
        newest = max(run["created_at"] for run in runs)
        early = [row for row in outcomes
                 if (parse_day(row["window_start"]) or "9999") < newest.strftime("%Y-%m-%d")]
        if early:
            problems.append(
                f"{len(early)} 条结果的开始日早于最新快照（{newest:%Y-%m-%d}）。"
                "打分发生在结果之后不构成预测 —— 回测时加 --after-run-only。"
            )

    print("\n=== 结论 ===")
    if problems:
        for index, problem in enumerate(problems, start=1):
            print(f"  {index}. ⚠️  {problem}")
    else:
        print("  ✅ 数据看起来够用。跑："
              f"python scripts/backtest.py --run {runs[0]['id']} "
              "--metric gross_profit --after-run-only")
    return 0


def main() -> int:
    args = parse_args()

    if args.template:
        print(TEMPLATE_HEADER)
        return 0

    service.prepare_db()

    if args.doctor:
        return cmd_doctor()
    if args.metrics:
        return cmd_metrics()
    if args.runs:
        return cmd_runs()

    if args.metric not in METRICS:
        print(f"❌ 未知指标 {args.metric!r}，可选：{', '.join(METRICS)}")
        return 1

    if args.compare:
        run_a, run_b = args.compare
        try:
            comparison = service.compare_backtests(
                run_a, run_b, metric=args.metric, top_ratio=args.top_ratio,
                after_run_only=args.after_run_only,
            )
        except KeyError as exc:
            print(f"❌ {exc}")
            return 1

        print_result(comparison.left, args.top)
        print(f"\n{'=' * 78}\n")
        print_result(comparison.right, args.top)
        print(f"\n{'=' * 78}")
        print(f"\n{comparison.summary()}")
        return 0

    if args.run is None:
        print("请指定 --run / --compare / --runs / --metrics / --template 之一。")
        print("例如：python scripts/backtest.py --run 1 --metric gross_profit")
        return 1

    try:
        result = service.backtest_run(
            args.run, metric=args.metric, top_ratio=args.top_ratio,
            after_run_only=args.after_run_only,
        )
    except KeyError as exc:
        print(f"❌ {exc}")
        return 1

    print_result(result, args.top)
    if result.sample_size < MIN_SAMPLE:
        print(f"\n（可比商品少于 {MIN_SAMPLE} 个时，ρ 的置信度很低，请谨慎下结论）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
