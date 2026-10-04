"""Streamlit 看板冒烟测试。

看板有 1200+ 行，之前完全没有测试 —— 一个未捕获异常就白屏，而且只有人打开页面才发现。
这里用 ``AppTest`` 把整个脚本跑一遍，确认所有页签都能渲染。

不测交互逻辑（那些归 service / db 的测试），只保证「页面能打开」。
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def test_app_renders_all_tabs_without_exception(temp_db):
    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()

    assert not app.exception, [exc.value for exc in app.exception]
    assert app.title[0].value == "🛒 AI 选品库"

    labels = [tab.label for tab in app.tabs]
    for expected in ("📊 选品榜单", "⚖️ 权重调参", "🎯 效果回测", "📈 数据概览"):
        assert expected in labels

    subheaders = [item.value for item in app.subheader]
    assert "效果回测" in subheaders


def test_backtest_tab_renders_with_data(temp_db):
    """有商品、快照和结果数据时，回测页也不能报错。"""
    from app import db, service
    from app.models import ProductIn

    products = db.bulk_upsert([
        ProductIn(title=f"商品{index}", price=100.0 + index, cost=40.0, heat=90.0 - index)
        for index in range(6)
    ])
    run = service.create_snapshot("基线", profile="balanced")
    for index, product in enumerate(products):
        service.record_outcome(product.id, "2024-03-01", "2024-03-31",
                               revenue=1000.0 - index * 100, cogs=0.0)

    app = AppTest.from_file(str(APP_PATH), default_timeout=90).run()

    assert not app.exception, [exc.value for exc in app.exception]
    assert any("可比商品" in metric.label for metric in app.metric)
    assert service.backtest_run(run["id"]).sample_size == 6
