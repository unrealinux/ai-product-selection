"""淘宝 A2A 选品拉取工具。

走的是淘宝官方公开的 **A2A（Agent2Agent）服务端**，不是本地那个被内测门槛拦着的
``taobao-native`` CLI，也不需要 appKey。

典型用法：

    # 0. 自检：确认 agent card 可达、列出它声明了哪些 skill
    python scripts/taobao_fetch.py --check

    # 1. 试拉取，只看结果不入库
    python scripts/taobao_fetch.py --queries "保温杯,降噪耳机" --preview

    # 2. 正式拉取 + 入库 + 打分
    python scripts/taobao_fetch.py --queries "保温杯,降噪耳机" --limit 30 \\
        --max-detail 20 --import --score

    # 3. 对 2-5 个商品做结构化对比（用 item-compare skill）
    python scripts/taobao_fetch.py --compare 1060199595825,973592811340

⚠️ 价格陷阱（本工具已内置防护）
    搜索页价格 ≠ 真实售价。实测同一保温杯：搜索返回 44.9，详情返回 79.9，差 78%。
    因此默认 **拿不到详情就丢弃该商品**，而不是退回去用搜索价 ——
    毛利率是权重最高的维度，用错价格整个排序就是错的。
    确实想保留时加 --allow-missing-detail，但 note 里会带 ⚠️ 标注。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import service  # noqa: E402
from app.config import settings  # noqa: E402
from app.sources.taobao import (  # noqa: E402
    SORT_OPTIONS,
    TaobaoA2AClient,
    TaobaoA2AError,
    TaobaoSource,
    TaobaoTaskFailed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从淘宝 A2A 拉取选品数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--queries", default="", metavar="关键词1,关键词2",
                        help="逗号分隔的搜索关键词，每个词一次召回")
    parser.add_argument("--limit", type=int, default=30, help="每个关键词召回多少候选（默认 30）")
    parser.add_argument("--sort", default="sales_desc", choices=list(SORT_OPTIONS),
                        help="召回排序（默认 sales_desc）")
    parser.add_argument("--detail-batch", type=int, default=None,
                        help="每批取几个详情（官方上限 10）")
    parser.add_argument("--max-detail", type=int, default=0,
                        help="最多为多少个商品取详情，控制调用次数（0=不限）")
    parser.add_argument("--category", default="", help="覆盖类目名（默认用搜索词）")

    parser.add_argument("--keep-ads", action="store_true", help="保留 isAd=true 的广告位商品")
    parser.add_argument("--allow-missing-detail", action="store_true",
                        help="⚠️ 拿不到详情时保留商品并用搜索页价格（会标注，但不建议）")
    parser.add_argument("--neutral-heat", action="store_true",
                        help="热度一律取 50，不用销量位次代理")
    parser.add_argument("--no-weight-extract", action="store_true",
                        help="不从 SKU 名/标题抽重量（一律用缺省值）")

    parser.add_argument("--check", action="store_true", help="自检：agent card 与声明的 skill")
    parser.add_argument("--compare", default="", metavar="ID1,ID2",
                        help="对 2-5 个商品做结构化对比（item-compare）")
    parser.add_argument("--preview", action="store_true", help="只预览，不写库（默认行为）")
    parser.add_argument("--import", dest="do_import", action="store_true", help="确认导入数据库")

    parser.add_argument("--score", action="store_true", help="导入后立即打分")
    parser.add_argument("--use-llm", action="store_true", help="打分时附带大模型点评")
    parser.add_argument("--top", type=int, default=10, help="打印前 N 条")
    return parser.parse_args()


def build_source(args: argparse.Namespace, client: TaobaoA2AClient) -> TaobaoSource:
    queries = [q.strip() for q in args.queries.replace("，", ",").split(",") if q.strip()]
    return TaobaoSource(
        queries,
        limit=args.limit,
        sort=args.sort,
        detail_batch=args.detail_batch,
        max_detail=args.max_detail,
        drop_ads=not args.keep_ads,
        require_detail=not args.allow_missing_detail,
        heat_mode="neutral" if args.neutral_heat else "rank",
        extract_weight=not args.no_weight_extract,
        category=args.category,
        client=client,
    )


def cmd_check(client: TaobaoA2AClient) -> int:
    print(f"接口地址：{client.base_url}")
    try:
        card = client.agent_card()
    except Exception as exc:  # noqa: BLE001 - 自检要把各种失败都显示出来
        print(f"❌ agent card 读取失败：{exc}", file=sys.stderr)
        return 1

    print(f"✅ agent card 可达")
    print(f"   name        : {card.get('name')}")
    print(f"   description : {card.get('description')}")
    print(f"   provider    : {(card.get('provider') or {}).get('organization')}")
    print(f"   version     : {card.get('version')}")
    interfaces = card.get("supportedInterfaces") or []
    for item in interfaces:
        print(f"   interface   : {item.get('url')} "
              f"({item.get('protocolBinding')} v{item.get('protocolVersion')})")
    print("\n声明的 skills：")
    for skill in card.get("skills") or []:
        print(f"   {skill.get('id'):<14} {skill.get('name')}  —— {skill.get('description')}")
    print("\n提示：本地 taobao-native CLI 是另一套东西（DOM 自动化），当前账号被内测门槛拦着；")
    print("      本工具走的是上面这个 A2A 接口，无需内测资格。")
    return 0


def cmd_compare(client: TaobaoA2AClient, raw: str) -> int:
    ids = [item.strip() for item in raw.replace("，", ",").split(",") if item.strip()]
    try:
        result = client.compare(ids, query="哪个更值得买")
    except (TaobaoA2AError, ValueError) as exc:
        print(f"对比失败：{exc}", file=sys.stderr)
        return 1

    print(f"artifact: {result.artifact}")
    if not result.data:
        print("服务端只返回了渲染卡片，没有结构化数据。")
        print("（item-compare 的对比结果由前端渲染，这里拿不到明细是正常的）")
        return 0
    import json

    print(json.dumps(result.data, ensure_ascii=False, indent=2)[:3000])
    return 0


def main() -> int:
    args = parse_args()
    client = TaobaoA2AClient.from_settings()

    if args.check:
        return cmd_check(client)
    if args.compare:
        return cmd_compare(client, args.compare)

    if not args.queries.strip():
        print("需要 --queries 指定搜索关键词（或先用 --check 自检）", file=sys.stderr)
        return 2

    source = build_source(args, client)
    if not source.require_detail:
        print("⚠️  已关闭「必须拿到详情」的保护，搜索页价格会被当作售价使用。", file=sys.stderr)

    print(f"拉取中：关键词={source.queries} 排序={source.sort} 每词上限={source.limit}")
    try:
        products = source.fetch()
    except TaobaoTaskFailed as exc:
        print(f"\nA2A 任务失败：{exc.message or exc}", file=sys.stderr)
        return 1
    except TaobaoA2AError as exc:
        print(f"\nA2A 调用失败：{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"\n参数错误：{exc}", file=sys.stderr)
        return 2

    print(f"\n{source.report.summary()}")
    for warning in source.report.warnings:
        print(f"  ⚠️  {warning}")
    for note in source.report.notes:
        print(f"  ℹ️  {note}")

    if not products:
        print("\n没有拿到任何商品。")
        return 0

    if args.preview:
        print(f"\n{'#':<4}{'售价':<10}{'热度':<8}{'重量kg':<9}{'类目':<12}{'商品'}")
        print("-" * 96)
        for index, item in enumerate(products[: args.top], start=1):
            print(f"{index:<6}{item.price:<12.2f}{item.heat:<10.1f}"
                  f"{item.weight_kg:<11.2f}{item.category[:10]:<14}{item.title[:34]}")
        print(f"\n示例来源说明：\n  {products[0].note.split('｜数据说明：')[-1]}")
        print("\n（--preview 模式，未写入数据库。确认无误后加 --import）")
        return 0

    service.prepare_db()
    saved = service.import_products(products)
    print(f"\n已写入数据库 {len(saved)} 条 → {settings.db_path}")

    if args.score:
        results = service.score_all(use_llm=args.use_llm)
        print(f"\n{'排名':<4}{'总分':<8}{'等级':<6}{'商品'}")
        print("-" * 68)
        for index, item in enumerate(results[: args.top], start=1):
            print(f"{index:<6}{item['total']:<10}{item['grade']:<8}{item['title'][:36]}")
        print(f"\n完成，共打分 {len(results)} 个商品。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
