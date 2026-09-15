"""Streamlit 看板：浏览榜单、录入商品、触发打分。

启动：
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app import __version__, crawler, db, llm, service
from app.config import DIMENSION_LABELS, settings
from app.enrich import PROVENANCE_SEP, enrich
from app.models import ProductIn
from app.sources.douyin import (
    FIELD_PROVENANCE,
    MAX_PAGE_SIZE,
    DouyinAPIError,
    DouyinConfigError,
    DouyinSource,
)
from app.weights import DIMENSIONS, PRESETS, describe, normalize, preset_weights

st.set_page_config(page_title="AI 选品库", page_icon="🛒", layout="wide")

GRADE_EMOJI = {"S": "🏆", "A": "🥇", "B": "🥈", "C": "🥉", "D": "⛔"}

#: 榜单里各维度得分的列名。margin 单列出来是为了不和真实的「毛利率」百分比撞名
DIMENSION_COLUMNS = {**DIMENSION_LABELS, "margin": "毛利得分"}

#: 抖音接口 search_type 取值（官方文档 api-docs/61/1725）
SEARCH_TYPE_LABELS = {
    0: "默认排序",
    1: "历史销量",
    2: "价格",
    3: "佣金金额",
    4: "佣金比例",
}


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
    for key, label in DIMENSION_COLUMNS.items():
        dims = frame["dimensions"].map(lambda d: d.get(key, 0.0))
        frame[label] = dims.round(1)
    columns = ["title", "category", "price", "cost", "毛利率", "total", "等级",
               *DIMENSION_COLUMNS.values(), "advice", "llm_review"]
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
        width="stretch",
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


DOUYIN_ENV_SNIPPET = """APS_DOUYIN_APP_KEY=你的app_key
APS_DOUYIN_APP_SECRET=你的app_secret
APS_DOUYIN_SHOP_ID=你的shop_id
APS_DOUYIN_ACCESS_TOKEN=换取后填入
"""


def render_provenance() -> None:
    """把「哪些维度来自接口、哪些是缺省值」摆在明面上。"""
    with st.expander("数据可信度：哪些维度来自接口，哪些是缺省值"):
        st.dataframe(
            pd.DataFrame([{"维度": key, "来源": value} for key, value in FIELD_PROVENANCE.items()]),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "重量 / 复购 / 合规三项接口未提供。不补齐时它们取缺省值，抖音来源商品的总分"
            "主要由销量与佣金率驱动；勾选「补齐重量/复购/合规」后会由大模型估算，"
            "但**仍是估算值**，需人工复核后再依赖排序。"
        )


def render_douyin_error_hint(exc: DouyinAPIError) -> None:
    """把官方返回码翻译成可执行的下一步。"""
    if exc.expired_token:
        st.info(
            "access_token 已失效。执行 `python scripts/douyin_fetch.py --token` "
            "换取新 token 后更新 .env 的 `APS_DOUYIN_ACCESS_TOKEN`。"
        )
    elif exc.code == 9:
        st.info("触发平台限流，请减少翻页数 / 每页条数，或稍后重试。")
    elif exc.code in {30001, 30004}:
        st.info("认证失败。请核对 .env 中 `APS_DOUYIN_APP_KEY` / `APS_DOUYIN_APP_SECRET`。")
    elif exc.code == 11:
        st.info("签名校验失败。请确认 `APS_DOUYIN_SIGN_METHOD` 与平台应用配置一致。")
    elif exc.code in {40004, 50002}:
        st.info("业务参数被拒绝。请检查排序方式、佣金率区间、类目 ID 是否合法。")


def douyin_preview_frame(products: list[ProductIn]) -> pd.DataFrame:
    """拉取结果预览表。"""
    rows = []
    for item in products:
        margin = (item.price - item.cost) / item.price if item.price else 0.0
        note = item.note or ""
        provenance = note.split(PROVENANCE_SEP, 1)[1] if PROVENANCE_SEP in note else ""
        rows.append({
            "商品": item.title,
            "类目": item.category,
            "售价(元)": item.price,
            "商家实收(元)": item.cost,
            "佣金率": f"{margin:.1%}",
            "重量(kg)": item.weight_kg,
            "复购潜力": round(item.repurchase, 1),
            "合规风险": round(item.compliance_risk, 1),
            "需求热度": round(item.heat, 1),
            "竞争度": round(item.competition, 1),
            "传播潜力": round(item.virality, 1),
            "数据说明": provenance,
            "链接": item.url,
        })
    return pd.DataFrame(rows)


def page_douyin() -> None:
    st.subheader("抖音精选联盟拉取")
    st.caption(
        "官方 API `buyin.kolMaterialsProductsSearch` ｜ "
        "文档 https://op.jinritemai.com/docs/api-docs/61/1725"
    )

    if not settings.douyin_ready:
        st.warning("未检测到抖音凭据，暂时无法拉取。")
        with st.expander("如何配置（三步）", expanded=True):
            st.markdown(
                "**1. 申请应用** —— 到 https://op.jinritemai.com/ 创建应用（自用型即可），"
                "取得 `app_key` / `app_secret`，并开通精选联盟相关权限。"
            )
            st.markdown("**2. 填写 `.env`**")
            st.code(DOUYIN_ENV_SNIPPET, language="dotenv")
            st.markdown("**3. 换取 access_token** —— 填好前两项与 shop_id 后执行：")
            st.code("python scripts/douyin_fetch.py --token", language="bash")
            st.caption("把返回的 access_token 写入 `.env` 的 `APS_DOUYIN_ACCESS_TOKEN`，然后刷新本页。")
        render_provenance()
        return

    st.success(
        f"已配置：app_key `{settings.douyin_app_key[:6]}…` ｜ "
        f"shop_id `{settings.douyin_shop_id or '未设置'}` ｜ "
        f"签名 `{settings.douyin_sign_method}`"
    )

    with st.form("douyin_fetch_form"):
        keywords = st.text_input(
            "关键词（逗号分隔；留空则按全量召回，竞争度会失真）",
            value="", placeholder="例如：咖啡,保温杯",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            pages = st.number_input("每个关键词翻页数", 1, 20, 1, step=1)
            page_size = st.number_input(
                f"每页条数（官方上限 {MAX_PAGE_SIZE}）", 1, MAX_PAGE_SIZE, MAX_PAGE_SIZE, step=1
            )
        with col2:
            search_type = st.selectbox(
                "召回排序", options=list(SEARCH_TYPE_LABELS),
                format_func=lambda key: SEARCH_TYPE_LABELS[key], index=1,
            )
            order = st.radio("排序方向", ["降序", "升序"], horizontal=True)
        with col3:
            cos_ratio_min_pct = st.number_input("最低佣金率（%，0=不限）", 0.0, 80.0, 0.0, step=0.5)
            first_cids_raw = st.text_input("一级类目 ID（逗号分隔，可留空）", value="")

        col4, col5, col6 = st.columns(3)
        with col4:
            only_in_stock = st.checkbox("仅保留在售商品", value=True)
        with col5:
            score_now = st.checkbox("导入后立即打分", value=True)
        with col6:
            use_llm = st.checkbox("启用大模型点评", value=False, disabled=not llm.is_available())

        col7, col8 = st.columns(2)
        with col7:
            enrich_llm = st.checkbox(
                "补齐重量/复购/合规", value=False, disabled=not llm.is_available(),
                help="接口不返回这三个字段，用大模型按标题+类目估算。结果会写进数据说明。",
            )
            enrich_judge = st.checkbox(
                "让大模型判断热度/传播力/类目", value=False,
                disabled=not llm.is_available(),
                help="传播力原本只是佣金率代理，改由大模型判断；"
                     "顺带把「抖音类目-2634」这类占位值换成可读类目名。",
            )
        with col8:
            override_heat = st.checkbox(
                "允许覆盖接口热度", value=False, disabled=not llm.is_available(),
                help="需求热度由接口的真实销量推导，默认不覆盖 —— 大模型判断通常不如销量可靠。",
            )
            enrich_detail_limit = st.number_input(
                "额外用商品详情接口补重量（仅自己店铺的商品有效，0=不试）",
                0, 100, 0, step=5,
            )

        submitted = st.form_submit_button("开始拉取", type="primary")

    if submitted:
        keywords_list = [item.strip() for item in keywords.split(",") if item.strip()]
        first_cids = [int(c) for c in first_cids_raw.replace("，", ",").split(",")
                      if c.strip().isdigit()]
        source = DouyinSource(
            keywords_list,
            page_size=int(page_size),
            max_pages=int(pages),
            search_type=int(search_type),
            sort_type=0 if order == "升序" else 1,
            first_cids=first_cids or None,
            cos_ratio_min=int(cos_ratio_min_pct * 100) or None,
            only_in_stock=only_in_stock,
        )
        try:
            with st.spinner("正在调用抖店开放平台…"):
                products = source.fetch()
        except DouyinConfigError as exc:
            st.error(f"配置错误：{exc}")
            return
        except DouyinAPIError as exc:
            st.error(f"接口调用失败：{exc}")
            render_douyin_error_hint(exc)
            return

        st.session_state.pop("douyin_enrich", None)
        if products and (enrich_llm or enrich_judge or enrich_detail_limit):
            try:
                with st.spinner("正在补齐重量 / 复购 / 合规…"):
                    products, enrich_report = enrich(
                        products,
                        client=source.client if enrich_detail_limit else None,
                        use_llm=enrich_llm or enrich_judge,
                        detail_limit=int(enrich_detail_limit),
                        judge=enrich_judge,
                        override_heat=override_heat,
                    )
            except Exception as exc:  # noqa: BLE001 - 补齐失败不应弄丢已拉到的数据
                st.warning(f"维度补齐失败，保留原始数据：{exc}")
            else:
                st.session_state["douyin_enrich"] = {
                    "summary": enrich_report.summary(),
                    "notes": list(enrich_report.notes),
                    "errors": list(enrich_report.errors),
                }

        st.session_state["douyin_products"] = products
        if not products:
            st.session_state["douyin_flash"] = (
                "info", "没有拉到商品。建议填写关键词缩小范围，或放宽佣金率限制。"
            )
            st.rerun()
        elif not score_now:
            st.session_state["douyin_flash"] = ("info", f"已拉取 {len(products)} 条（未入库）。")
            st.rerun()
        else:
            saved = service.import_products(products)
            results = service.score_all(use_llm=use_llm)
            st.session_state["douyin_flash"] = (
                "success",
                f"已拉取 {len(products)} 条，入库 {len(saved)} 条，完成打分 {len(results)} 条。",
            )
            st.rerun()

    flash = st.session_state.pop("douyin_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

    enrich_info = st.session_state.get("douyin_enrich")
    if enrich_info:
        st.info(f"维度补齐：{enrich_info['summary']}")
        for note in enrich_info["notes"]:
            st.caption(f"注：{note}")
        for error in enrich_info["errors"]:
            st.caption(f"⚠️ {error}")

    products = st.session_state.get("douyin_products") or []
    if products:
        st.markdown(f"#### 拉取结果（{len(products)} 条）")
        frame = douyin_preview_frame(products)
        st.dataframe(frame, width="stretch", hide_index=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "导出 CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                file_name="douyin_products.csv", mime="text/csv",
            )
        with col_b:
            if st.button("清空本次结果"):
                st.session_state.pop("douyin_products", None)
                st.session_state.pop("douyin_enrich", None)
                st.rerun()
        st.caption(
            "「商家实收」= 售价 − 达人佣金；「佣金率」= 达人佣金 / 售价。"
            "这是分销带货视角的口径，自营请自行覆盖成本。右侧「数据说明」标明每个维度的实际来源。"
        )
    render_provenance()
    st.divider()
    st.caption(
        "也可走命令行：`python scripts/douyin_fetch.py --keywords \"咖啡\" --pages 2 --dry-run`"
        " ｜ `--check` 打印配置状态与维度来源"
    )


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
        }), width="stretch", hide_index=True)

    st.divider()
    st.markdown("**当前全局权重**")
    st.dataframe(
        pd.DataFrame([
            {"维度": DIMENSION_LABELS.get(name, name),
             "权重": f"{normalize(settings.weights)[name]:.1%}"}
            for name in DIMENSIONS
        ]),
        width="stretch", hide_index=True,
    )

    st.divider()
    st.caption(
        f"AI 选品库 v{__version__} ｜ 数据库：`{settings.db_path}` ｜ "
        f"LLM：{'已启用 ' + settings.llm_model if llm.is_available() else '未启用（纯规则打分）'}"
    )


# --------------------------------------------------------------------------- #
# 权重调参与 A/B 对比
# --------------------------------------------------------------------------- #

def _load_profiles() -> list[dict]:
    return db.list_profiles()


def flash(kind: str, message: str) -> None:
    """记录一条提示，留到 rerun 之后展示。

    直接调 ``st.success()`` 再 ``st.rerun()`` 会把消息丢掉 —— 重跑会清空当前渲染。
    """
    st.session_state.setdefault("_flashes", []).append((kind, message))


def render_flashes() -> None:
    """展示并清空上一轮留下的提示。"""
    for kind, message in st.session_state.pop("_flashes", []):
        getattr(st, kind)(message)


def tuning_weights() -> None:
    """① 权重方案：调滑块、存方案。"""
    preset_options = ["（不载入）", *PRESETS]
    chosen = st.selectbox(
        "从内置预设载入", preset_options,
        format_func=lambda key: "（不载入）" if key == "（不载入）"
        else f"{PRESETS[key]['label']} —— {PRESETS[key]['description']}",
    )
    if chosen != "（不载入）" and st.button(f"应用预设「{PRESETS[chosen]['label']}」"):
        st.session_state["tuning_weights"] = preset_weights(chosen)
        st.rerun()

    current = st.session_state.get("tuning_weights") or settings.weights
    st.markdown("**调整权重**（保存时会自动归一化到 100%）")

    raw: dict[str, float] = {}
    columns = st.columns(4)
    for index, name in enumerate(DIMENSIONS):
        with columns[index % 4]:
            raw[name] = st.slider(
                DIMENSION_LABELS.get(name, name), 0.0, 1.0,
                float(current.get(name, 0.0)), 0.01, key=f"tune_{name}",
            )

    normalized = normalize(raw)
    st.caption(f"归一化后重心：{describe(normalized)}")
    st.dataframe(
        pd.DataFrame([
            {"维度": DIMENSION_LABELS.get(name, name),
             "权重": f"{normalized[name]:.1%}",
             "数值": round(normalized[name], 4)}
            for name in DIMENSIONS
        ]),
        width="stretch", hide_index=True,
    )

    with st.form("save_profile_form"):
        col1, col2 = st.columns([1, 2])
        name = col1.text_input("方案名", placeholder="例如：毛利优先-自用")
        description = col2.text_input("描述", placeholder="为什么这么调")
        if st.form_submit_button("保存为方案", type="primary"):
            try:
                profile = service.save_profile(name, raw, description)
            except ValueError as exc:
                st.error(str(exc))
            else:
                flash("success", f"已保存 #{profile['id']}「{profile['name']}」："
                      f"{describe(profile['weights'])}")
                st.rerun()

    st.divider()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("**已保存的方案**")
    with col_b:
        if st.button("写入内置预设"):
            flash("success", f"已写入 {service.install_presets()} 个预设")
            st.rerun()

    profiles = _load_profiles()
    if not profiles:
        st.info("还没有保存任何方案。")
        return
    st.dataframe(
        pd.DataFrame([
            {"ID": p["id"], "名称": p["name"],
             "重心": describe(p["weights"]), "描述": p["description"]}
            for p in profiles
        ]),
        width="stretch", hide_index=True,
    )
    to_delete = st.selectbox("删除方案", ["（不删除）", *[p["name"] for p in profiles]])
    if to_delete != "（不删除）" and st.button(f"确认删除「{to_delete}」"):
        db.delete_profile(to_delete)
        flash("success", f"已删除方案「{to_delete}」")
        st.rerun()


def tuning_snapshots() -> None:
    """② 打分快照：把一次打分固化成可对比的基线。"""
    st.caption(
        "快照会同时存下当时的**权重**与每个商品的**各维度得分**。"
        "因此后续再导数据、再改权重，都不会影响历史快照 —— 这是做 A/B 的前提。"
    )

    profiles = _load_profiles()
    source_options = ["当前滑块权重"] + [f"方案：{p['name']}" for p in profiles]
    source = st.radio("权重来源", source_options, horizontal=True)

    with st.form("create_run_form"):
        col1, col2 = st.columns([1, 2])
        label = col1.text_input("快照名称", value=f"运行 {datetime.now():%m-%d %H:%M}")
        note = col2.text_input("备注", placeholder="例如：heat 来自接口 sales")
        if st.form_submit_button("打分并固化快照", type="primary"):
            profile = source.split("：", 1)[1] if source.startswith("方案：") else None
            weights = None if profile else (st.session_state.get("tuning_weights")
                                            or settings.weights)
            try:
                run = service.create_snapshot(label, weights=weights,
                                              profile=profile, note=note)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
            else:
                flash("success", f"已创建快照 #{run['id']}「{run['label']}」："
                      f"{run['product_count']} 个商品，平均分 {run['avg_score']}")
                st.rerun()

    st.divider()
    st.markdown("**已有快照**")
    runs = service.list_snapshots(limit=100)
    if not runs:
        st.info("还没有快照。")
        return
    st.dataframe(
        pd.DataFrame([
            {"ID": r["id"], "名称": r["label"], "商品数": r["product_count"],
             "平均分": r["avg_score"], "重心": describe(r["weights"]),
             "创建时间": r["created_at"].strftime("%m-%d %H:%M"), "备注": r["note"]}
            for r in runs
        ]),
        width="stretch", hide_index=True,
    )


def _render_comparison(result) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("秩相关 Spearman ρ", f"{result.spearman:.4f}")
    col2.metric("平均排名变动", f"{result.avg_abs_rank_delta:.1f} 位")
    col3.metric("最大排名变动", f"{result.max_rank_delta} 位")

    st.info(f"{result.verdict}。（共同商品 {result.common} 个）")
    if result.notes:
        for note in result.notes:
            st.caption(f"注：{note}")

    if result.top_overlap:
        st.markdown("**Top-N 榜单重合度**")
        st.dataframe(
            pd.DataFrame([
                {"Top-N": f"Top{n}", "重合度": f"{value:.0%}", "数值": value}
                for n, value in result.top_overlap.items()
            ]),
            width="stretch", hide_index=True,
        )

    changed = [m for m in result.movers if m.rank_delta != 0]
    st.markdown(f"**排名发生变动的商品（{len(changed)} 个）**")
    if changed:
        st.dataframe(
            pd.DataFrame([
                {"商品": m.title, "类目": m.category, "方向": m.direction,
                 "位次变动": abs(m.rank_delta), "排名": f"#{m.rank_a} → #{m.rank_b}",
                 "总分": f"{m.total_a} → {m.total_b}"}
                for m in changed
            ]),
            width="stretch", hide_index=True,
        )
    else:
        st.success("没有任何商品发生排名变动 —— 两套权重得出了完全相同的排序。")

    with st.expander("权重差异明细"):
        st.dataframe(
            pd.DataFrame([
                {"维度": row["label"], "A": f"{row['a']:.1%}",
                 "B": f"{row['b']:.1%}", "变化": f"{row['delta']:+.1%}"}
                for row in result.weight_diff
            ]),
            width="stretch", hide_index=True,
        )


def tuning_compare() -> None:
    """③ A/B 对比。"""
    runs = service.list_snapshots(limit=100)
    if len(runs) < 2:
        st.info("至少需要两个快照才能对比。到「② 打分快照」页签创建。")
        return

    labels = [f"#{r['id']} {r['label']}" for r in runs]
    col1, col2, col3 = st.columns([2, 2, 1])
    label_a = col1.selectbox("A（基线）", labels, index=1)
    label_b = col2.selectbox("B（对照）", labels, index=0)
    top_n = col3.multiselect("Top-N", [5, 10, 20, 50], default=[10])

    if st.button("开始对比", type="primary"):
        try:
            st.session_state["comparison"] = service.compare_snapshots(
                runs[labels.index(label_a)]["id"],
                runs[labels.index(label_b)]["id"],
                movers=50,
            )
            if top_n:
                st.session_state["comparison"].top_overlap = {
                    n: value for n, value in
                    st.session_state["comparison"].top_overlap.items() if n in top_n
                }
        except KeyError as exc:
            st.error(str(exc))

    result = st.session_state.get("comparison")
    if result is None:
        st.caption("选好 A / B 后点「开始对比」。")
        return

    st.divider()
    st.markdown(
        f"**A** `#{result.run_a['id']} {result.run_a['label']}` "
        f"平均分 {result.run_a['avg_score']}　vs　"
        f"**B** `#{result.run_b['id']} {result.run_b['label']}` "
        f"平均分 {result.run_b['avg_score']}"
    )
    _render_comparison(result)


def tuning_sensitivity() -> None:
    """④ 维度影响力：哪个维度在真正决定排序。"""
    st.caption(
        "把某个维度的权重归零后重算排名，与原排名求相关。"
        "**ρ 越低说明该维度影响力越大**；若某维度归零后 ρ≈1，说明它几乎不参与决策，"
        "那么它的取值质量（接口值 or 大模型估算）也就无关紧要。"
    )
    runs = service.list_snapshots(limit=100)
    if not runs:
        st.info("还没有快照。到「② 打分快照」页签创建一个。")
        return

    col1, col2 = st.columns([3, 1])
    labels = [f"#{r['id']} {r['label']}" for r in runs]
    chosen = col1.selectbox("快照", labels)
    top_n = col2.number_input("Top-N", 1, 100, 10, step=5)

    run = runs[labels.index(chosen)]
    impacts = service.snapshot_sensitivity(run["id"], top_n=int(top_n))
    if not impacts:
        st.warning(
            "该快照只有一个非零维度，把它归零后所有商品分数相同、排名无意义，"
            "因此没有可计算的影响力数据。请先用包含多个维度的权重创建快照。"
        )
        return

    st.dataframe(
        pd.DataFrame([
            {"维度": impact.label, "权重": f"{impact.weight:.1%}",
             "ρ": impact.spearman, "影响力": impact.influence,
             "最大变动": impact.max_rank_delta,
             f"Top{int(top_n)} 重合": f"{impact.top_overlap:.0%}",
             "结论": impact.verdict}
            for impact in impacts
        ]),
        width="stretch", hide_index=True,
        column_config={
            "ρ": st.column_config.NumberColumn("ρ", format="%.4f"),
            "影响力": st.column_config.ProgressColumn(
                "影响力", min_value=0.0, max_value=1.0, format="%.4f"
            ),
        },
    )

    strongest = impacts[0]
    st.markdown(f"**影响力最大：{strongest.label}**（ρ={strongest.spearman:.4f}）—— {strongest.verdict}")
    weak = [impact for impact in impacts if impact.spearman >= 0.99]
    if weak:
        names = "、".join(impact.label for impact in weak)
        st.warning(
            f"几乎不影响排序：**{names}**。"
            "这些维度的取值质量对结果影响极小，不必急着补齐或精修。"
        )


def page_tuning() -> None:
    st.subheader("权重调参与 A/B 对比")
    st.caption(
        "权重决定结论。这里可以调权重、固化快照、对比两次运行的差异，"
        "并看出**到底哪个维度在真正决定排序**。"
    )
    tabs = st.tabs(["① 权重方案", "② 打分快照", "③ A/B 对比", "④ 维度影响力"])
    render_flashes()
    with tabs[0]:
        tuning_weights()
    with tabs[1]:
        tuning_snapshots()
    with tabs[2]:
        tuning_compare()
    with tabs[3]:
        tuning_sensitivity()


def main() -> None:
    _bootstrap()
    st.title("🛒 AI 选品库")
    st.caption("规则引擎 + 大模型的多维度选品打分与排序")

    tabs = st.tabs(["📊 选品榜单", "📥 数据导入", "🎯 抖音拉取", "⚖️ 权重调参",
                    "➕ 商品录入", "📈 数据概览"])
    with tabs[0]:
        page_leaderboard()
    with tabs[1]:
        page_import()
    with tabs[2]:
        page_douyin()
    with tabs[3]:
        page_tuning()
    with tabs[4]:
        page_create()
    with tabs[5]:
        page_stats()


main()
