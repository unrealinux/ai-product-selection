"""权重调参与 A/B 对比工具。

典型用法 —— 验证「用大模型覆盖热度值不值」：

    # 1. 先按接口热度打一个基线快照
    python scripts/douyin_fetch.py --keywords "咖啡" --enrich --score
    python scripts/tune_weights.py --run "基线-接口热度" --note "heat 来自接口 sales"

    # 2. 覆盖热度后重新导入打分，再存一个快照
    python scripts/douyin_fetch.py --keywords "咖啡" --enrich-judge --override-heat --score
    python scripts/tune_weights.py --run "对照-大模型热度" --note "heat 由大模型覆盖"
    #    第一个快照已固化，因此商品数据被覆盖也不影响对比

    # 3. 对比两次运行
    python scripts/tune_weights.py --compare 1 2

    # 4. 看哪个维度在真正决定排序
    python scripts/tune_weights.py --sensitivity 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, service  # noqa: E402
from app.config import DIMENSION_LABELS, settings  # noqa: E402
from app.weights import (  # noqa: E402
    DIMENSIONS,
    PRESETS,
    describe,
    normalize,
    parse_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="权重调参与 A/B 对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--presets", action="store_true", help="列出内置预设与当前默认权重")
    parser.add_argument("--profiles", action="store_true", help="列出数据库里已保存的方案")
    parser.add_argument("--install-presets", action="store_true", help="把内置预设写入数据库")

    parser.add_argument("--save", metavar="NAME", help="保存一个权重方案")
    parser.add_argument("--weights", default="", metavar="K=V,...",
                        help='权重，如 "margin=0.4,demand=0.1"；维度可用中文名')
    parser.add_argument("--description", default="", help="方案描述")

    parser.add_argument("--run", metavar="LABEL", help="按指定权重打分并固化快照")
    parser.add_argument("--profile", default=None, help="使用已保存的方案名或内置预设名")
    parser.add_argument("--note", default="", help="快照备注")
    parser.add_argument("--runs", action="store_true", help="列出所有快照")

    parser.add_argument("--compare", nargs=2, type=int, metavar=("RUN_A", "RUN_B"),
                        help="对比两个快照")
    parser.add_argument("--sensitivity", type=int, metavar="RUN_ID",
                        help="查看某个快照下各维度对排序的影响力")
    parser.add_argument("--top", type=int, default=10, help="打印前 N 条（默认 10）")
    return parser.parse_args()


def print_weights(weights: dict[str, float], indent: str = "    ") -> None:
    normalized = normalize(weights)
    for name in DIMENSIONS:
        value = normalized.get(name, 0.0)
        bar = "█" * max(1, round(value * 40)) if value > 0 else ""
        print(f"{indent}{DIMENSION_LABELS.get(name, name):<8}{value:>7.2%}  {bar}")


def cmd_presets() -> int:
    print("内置预设：\n")
    for name, spec in PRESETS.items():
        print(f"  {name}  ——  {spec['label']}")
        print(f"    {spec['description']}")
        print(f"    重心：{describe(spec['weights'])}")
        print_weights(spec["weights"], indent="      ")
        print()
    print("当前全局默认权重：")
    print_weights(settings.weights)
    return 0


def cmd_profiles() -> int:
    profiles = db.list_profiles()
    if not profiles:
        print("库里还没有保存任何方案。可以执行 --install-presets 写入内置预设。")
        return 0
    print(f"{'ID':<5}{'名称':<20}{'重心':<34}描述")
    print("-" * 100)
    for profile in profiles:
        print(f"{profile['id']:<7}{profile['name']:<22}"
              f"{describe(profile['weights']):<36}{profile['description']}")
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    try:
        weights = parse_weights(args.weights)
    except ValueError as exc:
        print(f"权重解析失败：{exc}", file=sys.stderr)
        return 2
    if not weights:
        print("--save 需要配合 --weights 使用", file=sys.stderr)
        return 2
    try:
        profile = service.save_profile(args.save, weights, args.description)
    except ValueError as exc:
        print(f"权重校验失败：{exc}", file=sys.stderr)
        return 2
    print(f"已保存方案 #{profile['id']} {profile['name']}：{describe(profile['weights'])}")
    print_weights(profile["weights"])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    explicit: dict[str, float] = {}
    if args.weights:
        try:
            explicit = parse_weights(args.weights)
        except ValueError as exc:
            print(f"权重解析失败：{exc}", file=sys.stderr)
            return 2
    try:
        run = service.create_snapshot(
            args.run, weights=explicit or None, profile=args.profile, note=args.note
        )
    except (ValueError, KeyError) as exc:
        print(f"创建快照失败：{exc}", file=sys.stderr)
        return 2

    print(f"已创建快照 #{run['id']}「{run['label']}」"
          f"（{run['product_count']} 个商品，平均分 {run['avg_score']}）")
    print(f"权重重心：{describe(run['weights'])}")
    if args.note:
        print(f"备注：{args.note}")

    detail = service.snapshot_detail(run["id"])
    print(f"\n{'排名':<5}{'总分':<9}{'商品'}")
    print("-" * 70)
    for item in detail["items"][: args.top]:
        print(f"{item['rank_no']:<7}{item['total']:<11.2f}{item['title'][:40]}")
    return 0


def cmd_runs() -> int:
    runs = service.list_snapshots()
    if not runs:
        print("还没有任何快照。用 --run LABEL 创建一个。")
        return 0
    print(f"{'ID':<5}{'名称':<26}{'商品数':<8}{'平均分':<9}{'创建时间':<21}备注")
    print("-" * 110)
    for run in runs:
        stamp = run["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{run['id']:<7}{run['label'][:24]:<28}{run['product_count']:<10}"
              f"{run['avg_score']:<11}{stamp:<23}{run['note']}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left, right = args.compare
    try:
        result = service.compare_snapshots(left, right, movers=args.top)
    except KeyError as exc:
        print(f"对比失败：{exc}", file=sys.stderr)
        return 1

    print(f"A = #{result.run_a['id']} {result.run_a['label']}"
          f"   商品 {result.run_a['product_count']} 个，平均分 {result.run_a['avg_score']}")
    print(f"    重心：{describe(result.run_a['weights'])}")
    print(f"B = #{result.run_b['id']} {result.run_b['label']}"
          f"   商品 {result.run_b['product_count']} 个，平均分 {result.run_b['avg_score']}")
    print(f"    重心：{describe(result.run_b['weights'])}")
    print()

    print("权重差异：")
    for row in result.weight_diff[:5]:
        arrow = "↑" if row["delta"] > 0 else ("↓" if row["delta"] < 0 else "=")
        print(f"  {row['label']:<8}{row['a']:>7.2%} → {row['b']:>7.2%}  {arrow} {abs(row['delta']):.2%}")

    print()
    print("对比结论：")
    print(f"  {result.summary()}")
    for note in result.notes:
        print(f"  注：{note}")

    moved = [mover for mover in result.movers if mover.rank_delta != 0]
    if moved:
        print(f"\n排名发生变动的商品（共 {len(moved)} 个）：")
        print(f"  {'方向':<6}{'位次':<7}{'商品':<28}{'排名变化':<18}{'总分变化'}")
        print("  " + "-" * 88)
        for mover in moved:
            rank_text = f"#{mover.rank_a} → #{mover.rank_b}"
            print(f"  {mover.direction:<8}{abs(mover.rank_delta):<9}"
                  f"{mover.title[:26]:<30}{rank_text:<20}"
                  f"{mover.total_a} → {mover.total_b}")
    else:
        print("\n没有任何商品发生排名变动 —— 两套权重得出了完全相同的排序。")

    print()
    if result.spearman >= 0.90:
        print("结论：两套权重下排序高度一致 —— 换权重的收益有限，不必在这上面纠结。")
    elif result.spearman >= 0.75:
        print("结论：排序有明显差异，建议结合业务目标挑选权重，而不是追求唯一最优。")
    else:
        print("结论：排序差异很大，权重选择会直接改变结论，务必用业务结果校准。")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    try:
        detail = service.snapshot_detail(args.sensitivity)
        impacts = service.snapshot_sensitivity(args.sensitivity, top_n=args.top)
    except KeyError as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 1

    print(f"快照 #{detail['id']}「{detail['label']}」（{detail['product_count']} 个商品）")
    print(f"权重重心：{describe(detail['weights'])}")
    print(f"\n各维度影响力（把该维度权重归零后，与原排名的相关性 ρ）：")
    if not impacts:
        print("  该快照只有一个非零维度，把它归零后所有商品分数相同、排名无意义，")
        print("  因此没有可计算的影响力数据。请先用包含多个维度的权重创建快照。")
        return 0
    print(f"  {'维度':<9}{'权重':<9}{'ρ':<10}{'影响力':<10}{'最大变动':<10}{'TopN 重合':<11}结论")
    print("  " + "-" * 88)
    for impact in impacts:
        print(f"  {impact.label:<11}{impact.weight:<11.2%}{impact.spearman:<12.4f}"
              f"{impact.influence:<12.4f}{impact.max_rank_delta:<12}"
              f"{impact.top_overlap:<13.0%}{impact.verdict}")

    print()
    if impacts:
        top = impacts[0]
        print(f"影响力最大：{top.label}（ρ={top.spearman:.4f}）—— {top.verdict}")
        weak = [item for item in impacts if item.spearman >= 0.99]
        if weak:
            names = "、".join(item.label for item in weak)
            print(f"几乎不影响排序：{names}")
            print("  → 这些维度的取值质量对结果影响极小，不必急着补齐或精修。")
    return 0


def main() -> int:
    args = parse_args()
    service.prepare_db()

    if args.presets:
        return cmd_presets()
    if args.profiles:
        return cmd_profiles()
    if args.install_presets:
        print(f"已写入 {service.install_presets()} 个内置预设。")
        return cmd_profiles()
    if args.save:
        return cmd_save(args)
    if args.run:
        return cmd_run(args)
    if args.runs:
        return cmd_runs()
    if args.compare:
        return cmd_compare(args)
    if args.sensitivity:
        return cmd_sensitivity(args)

    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
