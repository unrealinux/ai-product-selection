"""把示例数据导入数据库并完成打分。

用法：
    python scripts/seed_data.py                 # 纯规则打分
    python scripts/seed_data.py --use-llm       # 附加大模型点评（需配置 .env）
    python scripts/seed_data.py --reset         # 先清空数据库再导入
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import crawler, db, llm, service  # noqa: E402
from app.config import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入示例选品数据并打分")
    parser.add_argument("--path", default=str(crawler.SAMPLE_PATH), help="数据文件路径")
    parser.add_argument("--use-llm", action="store_true", help="启用大模型点评")
    parser.add_argument("--reset", action="store_true", help="导入前清空数据库")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.reset:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(str(settings.db_path) + suffix).unlink(missing_ok=True)
        print(f"已清空数据库：{settings.db_path}")

    service.prepare_db()

    products = crawler.load_json(args.path)
    saved = service.import_products(products)
    print(f"导入商品 {len(saved)} 条 → {settings.db_path}")

    if args.use_llm and not llm.is_available():
        print("提示：未配置 APS_LLM_* ，本次仅使用规则打分")

    results = service.score_all(use_llm=args.use_llm)
    print(f"\n{'排名':<4}{'总分':<8}{'等级':<6}{'商品':<28}{'毛利率':<8}")
    print("-" * 60)
    for index, item in enumerate(results, start=1):
        margin = f"{item['profit_margin']:.0%}"
        print(f"{index:<6}{item['total']:<10}{item['grade']:<8}{item['title']:<30}{margin}")

    print(f"\n完成，共打分 {len(results)} 个商品。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
