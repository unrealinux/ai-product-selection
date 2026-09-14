"""抖音精选联盟选品拉取工具。

用法：
    # 1. 检查配置（不会发请求）
    python scripts/douyin_fetch.py --check

    # 2. 自用型应用换取 access_token（需要 shop_id）
    python scripts/douyin_fetch.py --token

    # 3. 试拉取，只看结果不入库
    python scripts/douyin_fetch.py --keywords "咖啡,保温杯" --dry-run

    # 4. 正式拉取并按历史销量降序导入 + 打分
    python scripts/douyin_fetch.py --keywords "咖啡,保温杯" --pages 2 --score
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import service  # noqa: E402
from app.config import settings  # noqa: E402
from app.enrich import enrich  # noqa: E402
from app.sources.douyin import (  # noqa: E402
    ERROR_CODES,
    FIELD_PROVENANCE,
    DouyinAPIError,
    DouyinClient,
    DouyinConfigError,
    DouyinSource,
)

SEARCH_TYPE_LABELS = {
    0: "默认排序",
    1: "历史销量",
    2: "价格",
    3: "佣金金额",
    4: "佣金比例",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从抖音精选联盟拉取选品数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--keywords", default="", help="逗号分隔的搜索关键词，留空表示全量")
    parser.add_argument("--pages", type=int, default=1, help="每个关键词翻多少页（默认 1）")
    parser.add_argument("--page-size", type=int, default=20, help="每页条数，官方上限 20")
    parser.add_argument("--search-type", type=int, default=1, choices=sorted(SEARCH_TYPE_LABELS),
                        help="召回排序条件：0 默认 1 历史销量 2 价格 3 佣金金额 4 佣金比例")
    parser.add_argument("--sort-type", type=int, default=1, choices=[0, 1], help="0 升序 1 降序")
    parser.add_argument("--cos-ratio-min", type=int, default=None,
                        help="最低佣金率，乘 100（如 10%% 传 1000）")
    parser.add_argument("--first-cids", default="", help="逗号分隔的一级类目 ID")

    parser.add_argument("--check", action="store_true", help="只检查配置与签名自检，不发业务请求")
    parser.add_argument("--token", action="store_true", help="换取 access_token（自用型应用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写入数据库")
    parser.add_argument("--score", action="store_true", help="导入后立即打分")
    parser.add_argument("--use-llm", action="store_true", help="打分时附带大模型点评")
    parser.add_argument("--top", type=int, default=10, help="打印前 N 条（默认 10）")

    parser.add_argument(
        "--enrich", action="store_true",
        help="用大模型估算接口缺失的重量/复购/合规（需已配置 APS_LLM_*）",
    )
    parser.add_argument(
        "--enrich-detail", type=int, default=0, metavar="N",
        help="额外用商品详情接口补重量，最多 N 个（仅对已授权店铺自己的商品有效）",
    )
    parser.add_argument("--enrich-batch", type=int, default=None, help="大模型估算的批大小")
    parser.add_argument("--refresh-estimates", action="store_true", help="忽略估算缓存重新估算")
    return parser.parse_args()


def print_config_status() -> None:
    print("=== 抖音数据源配置 ===")
    rows = [
        ("APS_DOUYIN_APP_KEY", settings.douyin_app_key),
        ("APS_DOUYIN_APP_SECRET", settings.douyin_app_secret),
        ("APS_DOUYIN_ACCESS_TOKEN", settings.douyin_access_token),
        ("APS_DOUYIN_SHOP_ID", settings.douyin_shop_id),
    ]
    for name, value in rows:
        shown = f"{value[:6]}…（已配置，长度 {len(value)}）" if value else "未配置"
        print(f"  {name:<26} {shown}")
    print(f"  {'APS_DOUYIN_BASE_URL':<26} {settings.douyin_base_url}")
    print(f"  {'APS_DOUYIN_SIGN_METHOD':<26} {settings.douyin_sign_method}")
    print(f"\n整体可用：{'是' if settings.douyin_ready else '否'}")
    if not settings.douyin_ready:
        print("  → 请在 .env 中补齐 app_key / app_secret / access_token（参考 .env.example）")

    print("\n=== 维度数据来源说明 ===")
    for field, desc in FIELD_PROVENANCE.items():
        print(f"  {field:<16} {desc}")
    print("\n  可用 --enrich 让大模型估算重量/复购/合规；")
    print("  --enrich-detail N 会额外尝试商品详情接口（仅对自己店铺的商品有效）。")


def main() -> int:
    args = parse_args()

    if args.check:
        print_config_status()
        return 0

    try:
        client = DouyinClient.from_settings()
    except DouyinConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    if args.token:
        try:
            data = client.create_self_token()
        except (DouyinConfigError, DouyinAPIError) as exc:
            print(f"换取 access_token 失败：{exc}", file=sys.stderr)
            return 1
        token = str(data.get("access_token") or "")
        print("换取成功。")
        print(f"  access_token : {token[:8]}…（长度 {len(token)}，完整值已省略）")
        print(f"  expires_in   : {data.get('expires_in')}")
        print("\n请把完整 access_token 写入 .env 的 APS_DOUYIN_ACCESS_TOKEN")
        return 0

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    first_cids = [int(c) for c in args.first_cids.split(",") if c.strip().isdigit()]

    source = DouyinSource(
        keywords,
        page_size=args.page_size,
        max_pages=args.pages,
        search_type=args.search_type,
        sort_type=args.sort_type,
        first_cids=first_cids or None,
        cos_ratio_min=args.cos_ratio_min,
        client=client,
    )

    order = SEARCH_TYPE_LABELS.get(args.search_type, str(args.search_type))
    print(f"拉取中：关键词={keywords or ['<全量>']} 排序={order} "
          f"{'降序' if args.sort_type else '升序'} 每词 {args.pages} 页 × {args.page_size} 条")

    try:
        products = source.fetch()
    except DouyinAPIError as exc:
        print(f"\n接口调用失败：{exc}", file=sys.stderr)
        if exc.expired_token:
            print("→ access_token 已失效，请执行：python scripts/douyin_fetch.py --token",
                  file=sys.stderr)
        if exc.code == 9:
            print("→ 触发限流，请降低 --pages / --page-size 后重试", file=sys.stderr)
        return 1
    except DouyinConfigError as exc:
        print(f"\n配置错误：{exc}", file=sys.stderr)
        return 2

    if not products:
        print("没有拉到任何商品。若使用全量查询，建议加上 --keywords 缩小范围。")
        return 0

    print(f"\n共获取 {len(products)} 个商品。")

    if args.enrich or args.enrich_detail:
        products, enrich_report = enrich(
            products,
            client=client if args.enrich_detail else None,
            use_llm=args.enrich,
            detail_limit=args.enrich_detail,
            batch_size=args.enrich_batch,
            use_cache=not args.refresh_estimates,
        )
        print(f"维度补齐：{enrich_report.summary()}")
        for note in enrich_report.notes:
            print(f"  注：{note}")
        for error in enrich_report.errors:
            print(f"  错误：{error}", file=sys.stderr)

    if args.dry_run:
        print(f"\n{'#':<4}{'售价':<10}{'毛利率':<10}{'重量kg':<10}{'复购':<8}{'合规':<8}{'商品'}")
        print("-" * 92)
        for index, item in enumerate(products[: args.top], start=1):
            margin = (item.price - item.cost) / item.price if item.price else 0
            print(f"{index:<6}{item.price:<12.2f}{margin:<12.0%}"
                  f"{item.weight_kg:<12.2f}{item.repurchase:<10.0f}"
                  f"{item.compliance_risk:<10.0f}{item.title[:24]}")
        print("\n（--dry-run 模式，未写入数据库）")
        return 0

    service.prepare_db()
    saved = service.import_products(products)
    print(f"已写入数据库 {len(saved)} 条 → {settings.db_path}")

    if args.score:
        results = service.score_all(use_llm=args.use_llm)
        print(f"\n{'排名':<4}{'总分':<8}{'等级':<6}{'商品'}")
        print("-" * 60)
        for index, item in enumerate(results[: args.top], start=1):
            print(f"{index:<6}{item['total']:<10}{item['grade']:<8}{item['title'][:30]}")
        print(f"\n完成，共打分 {len(results)} 个商品。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
