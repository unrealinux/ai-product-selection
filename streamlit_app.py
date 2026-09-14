"""Streamlit 看板：浏览榜单、录入商品、触发打分。

启动：
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import __version__, crawler, db, llm, service
from app.config import DIMENSION_LABELS, settings
from app.models import ProductIn
from app.scoring import grade_of

st.set_page_config(page_title="AI 选品库", page_icon="🛒", layout="wide")

GRADE_EMOJI = {"S": "🏆", "A": "🥇", "B": "🥈", "C": "🥉", "D": "⛔"}


@st.cache_resource
def _bootstrap() -> bool:
    service.prepare_db()
    return True


def load_leaderboard(limit: int, category: str | None) -> pd.DataFrame:
    rows = service.leaderboard(limit=limit, category=category)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["等级"] = frame["grade"].map(lambda g: f"{GRADE_EMOJI.get(g, '')}{g}")
    frame["毛利率"] = frame["profit_margin"].map(lambda v: f"{v:.0%}")
    for key, label in DIMENSION_LABELS.items():
        dims = frame["dimensions"].map(lambda d: d.get(key, 0.0))
        frame[label] = dims.round(1)
    columns = ["title", "category", "price", "cost", "毛利率", "total", "等级",
               *DIMENSION_LABELS.values(), "advice", "llm_review"]
    frame = frame[[col for col in columns if col in frame.columns]]
    return frame.rename(columns={
        "title": "商品", "category": "类目", "price": "售价",
        "cost": "成本", "total": "总分", "advice": "规则建议", "llm_review": "AI 点评",
    })


def page_leaderboard() -> None:
    st.subheader("选品榜单")
    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.slider("展示数量", 5, 200, 20, step=5)
    with col2:
        categories = ["全部"] + sorted({p.category for p in db.list_products(limit=1000)})
        category = st.selectbox("类目筛选", categories)

    frame = load_leaderboard(limit, None if category == "全部" else category)
    if frame.empty:
        st.info("暂无数据。请到「数据导入」页导入示例数据，或到「商品录入」页手工添加。")
        return

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "总分": st.column_config.ProgressColumn(
                "总分", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )
    st.download_button(
        "导出 CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="leaderboard.csv",
        mime="text/csv",
    )


def page_import() -> None:
    st.subheader("数据导入")
    st.caption(f"示例数据文件：`{settings.base_dir / 'data' / 'sample_products.json'}`")
    use_llm = st.checkbox("导入后启用大模型点评", value=False, disabled=not llm.is_available())
    if not llm.is_available():
        st.caption("未检测到 APS_LLM_* 配置，将仅使用规则打分。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("导入内置示例数据", type="primary"):
            products = service.import_products(crawler.load_json(crawler.SAMPLE_PATH))
            service.score_all(use_llm=use_llm)
            st.success(f"已导入 {len(products)} 条商品并完成打分。")
            st.rerun()
    with col2:
        uploaded = st.file_uploader("或上传 JSON / JSONL 文件", type=["json", "jsonl"])
        if uploaded is not None and st.button("导入上传文件"):
            import json

            text = uploaded.getvalue().decode("utf-8")
            records = json.loads(text) if text.strip().startswith("[") else [
                json.loads(line) for line in text.splitlines() if line.strip()
            ]
            items = [ProductIn(**record) for record in records]
            service.import_products(items)
            service.score_all(use_llm=use_llm)
            st.success(f"已导入 {len(items)} 条商品并完成打分。")
            st.rerun()


def page_create() -> None:
    st.subheader("商品录入")
    with st.form("create_product"):
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.text_input("商品标题 *")
            category = st.text_input("类目", value="未分类")
            source = st.text_input("来源", value="manual")
            url = st.text_input("商品链接", value="")
        with col2:
            price = st.number_input("售价（元）", min_value=0.0, value=99.0, step=1.0)
            cost = st.number_input("成本（元）", min_value=0.0, value=35.0, step=1.0)
            weight_kg = st.number_input("重量（kg）", min_value=0.0, value=0.5, step=0.1)
        with col3:
            heat = st.slider("需求热度", 0, 100, 60)
            competition = st.slider("竞争度（越低越好）", 0, 100, 50)
            repurchase = st.slider("复购潜力", 0, 100, 40)

        col4, col5 = st.columns(2)
        with col4:
            compliance_risk = st.slider("合规风险（越低越好）", 0, 100, 20)
        with col5:
            virality = st.slider("内容传播力", 0, 100, 50)

        note = st.text_area("备注", value="")
        submitted = st.form_submit_button("保存并打分", type="primary")

    if submitted:
        if not title.strip():
            st.error("商品标题不能为空。")
            return
        product = db.upsert_product(ProductIn(
            title=title, category=category or "未分类", price=price, cost=cost,
            source=source or "manual", url=url, heat=heat, competition=competition,
            weight_kg=weight_kg, repurchase=repurchase, compliance_risk=compliance_risk,
            virality=virality, note=note,
        ))
        result = service.score_product(product.id, use_llm=llm.is_available())
        if result:
            st.success(f"已保存 #{product.id}，总分 {result['total']}（{result['grade']}）")
            st.info(result["advice"])
            if result.get("llm_review"):
                st.info(f"AI 点评：{result['llm_review']}")


def page_stats() -> None:
    st.subheader("数据概览")
    data = service.dashboard_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("商品总数", data["total"])
    col2.metric("已打分", data["scored"])
    col3.metric("平均总分", data["avg_score"])

    if data["grade_distribution"]:
        st.bar_chart(pd.DataFrame(
            {"等级": list(data["grade_distribution"].keys()),
             "数量": list(data["grade_distribution"].values())}
        ).set_index("等级"))

    if data["top_categories"]:
        st.caption("各类目平均分")
        st.dataframe(pd.DataFrame(data["top_categories"]).rename(columns={
            "category": "类目", "n": "商品数", "avg_score": "平均分",
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        f"AI 选品库 v{__version__} ｜ 数据库：`{settings.db_path}` ｜ "
        f"LLM：{'已启用 ' + settings.llm_model if llm.is_available() else '未启用（纯规则打分）'}"
    )


def main() -> None:
    _bootstrap()
    st.title("🛒 AI 选品库")
    st.caption("规则引擎 + 大模型的多维度选品打分与排序")

    tabs = st.tabs(["📊 选品榜单", "📥 数据导入", "➕ 商品录入", "📈 数据概览"])
    with tabs[0]:
        page_leaderboard()
    with tabs[1]:
        page_import()
    with tabs[2]:
        page_create()
    with tabs[3]:
        page_stats()


main()
