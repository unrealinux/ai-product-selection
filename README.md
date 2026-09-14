# AI 选品库

> 用**规则引擎 + 大模型**对候选商品做多维度打分、排序与点评，帮你把「凭感觉选品」变成「看数据决策」。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

---

## 这个项目解决什么问题

选品的难点不是「找不到货」，而是**候选太多、维度太多、判断太主观**。同一个商品，运营看热度、财务看毛利、供应链看重量，最后往往拍脑袋决定。

AI 选品库把决策拆成两层：

| 层次 | 角色 | 特点 |
| --- | --- | --- |
| **规则引擎** | 确定性打分 | 7 个维度归一化到 0-100，按权重加权；结果可解释、可复现、可单测 |
| **大模型** | 定性点评 | 输出机会点/风险点与可执行建议，并允许 ±10 分的有限修正 |

> 关键设计：**大模型不能推翻规则结果**，只能做有限修正。这避免了 LLM 幻觉直接污染排序，让榜单始终可解释。

---

## 打分模型

总分 = 各维度得分 × 权重之和（权重可配置，见 `app/config.py`）。

| 维度 | 权重 | 计算方式 | 方向 |
| --- | --- | --- | --- |
| 毛利率 | 24% | `(售价-成本)/售价` 映射到 0〜60% → 0〜100 | 越高越好 |
| 需求热度 | 22% | 直接取值 | 越高越好 |
| 竞争度 | 18% | `100 - 竞争度` | 越低越好 |
| 内容传播力 | 12% | 直接取值 | 越高越好 |
| 物流友好 | 8% | 1kg 满分，5kg 及以上 0 分，线性插值 | 越轻越好 |
| 复购潜力 | 8% | 直接取值 | 越高越好 |
| 合规安全 | 8% | `100 - 合规风险` | 越低越好 |

等级划分：`S ≥ 85` ｜ `A ≥ 75` ｜ `B ≥ 65` ｜ `C ≥ 55` ｜ `D < 55`

规则引擎还会根据短板维度生成**可执行建议**，例如：

> 【可测款观察】毛利率仅 18%，低于健康线 25%，建议提价或压成本；竞争度 85 偏高，需要差异化卖点或细分人群切入。

---

## 快速开始

```bash
git clone https://github.com/unrealinux/ai-product-selection.git
cd ai-product-selection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # 不配置 LLM 也能跑，自动降级为纯规则打分

# 导入内置示例数据并打分
python scripts/seed_data.py
```

### 启动 Web API

```bash
uvicorn app.api:app --reload
# 打开 http://127.0.0.1:8000/docs 查看交互式文档
```

### 启动可视化看板

```bash
streamlit run streamlit_app.py
# 打开 http://localhost:8501
```

看板包含四个页签：**选品榜单 / 数据导入 / 商品录入 / 数据概览**，支持 CSV 导出。

---

## 接入大模型（可选）

任何 **OpenAI 兼容**接口都可以：OpenAI、DeepSeek、通义千问、本地 vLLM / Ollama。

```dotenv
APS_LLM_ENABLED=true
APS_LLM_BASE_URL=https://api.deepseek.com/v1
APS_LLM_API_KEY=sk-xxxxxxxx
APS_LLM_MODEL=deepseek-chat
```

启用后：

```bash
python scripts/seed_data.py --use-llm     # 导入 + 规则打分 + AI 点评
curl -X POST "http://127.0.0.1:8000/score?use_llm=true"
```

模型调用失败或未安装 `requests` 时会**静默降级**，不会中断主流程。

---

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，含 LLM 可用状态 |
| `GET` | `/products` | 商品列表，支持 `category` / `source` / `limit` |
| `GET` | `/products/{id}` | 商品详情 |
| `POST` | `/products` | 新增单个商品 |
| `POST` | `/import` | 从数据源批量导入，可选立即打分 |
| `POST` | `/score/{id}` | 单个商品打分 |
| `POST` | `/score` | 全库打分 |
| `GET` | `/leaderboard` | 选品榜单（按总分倒序） |
| `GET` | `/stats` | 看板统计 |

示例：

```bash
# 导入示例数据并打分
curl -X POST http://127.0.0.1:8000/import \
  -H "Content-Type: application/json" \
  -d '{"source": "sample", "score": true, "use_llm": false}'

# 查看榜单前 10
curl "http://127.0.0.1:8000/leaderboard?limit=10"
```

---

## 导入自己的数据

准备 JSON 数组或 JSONL 文件，字段如下（除 `title` 外均可省略）：

```json
[
  {
    "title": "便携式迷你折叠洗衣机",
    "category": "家居清洁",
    "price": 199,
    "cost": 78,
    "source": "1688",
    "url": "https://example.com/p/xxx",
    "heat": 78,
    "competition": 62,
    "weight_kg": 2.4,
    "repurchase": 15,
    "compliance_risk": 10,
    "virality": 82,
    "note": "宿舍场景，短视频演示效果好"
  }
]
```

```bash
python scripts/seed_data.py --path my_products.json          # 命令行导入
curl -X POST http://127.0.0.1:8000/import -d '{"source":"json","path":"my_products.json"}'
```

字段说明：

- `heat` / `competition` / `repurchase` / `compliance_risk` / `virality`：0〜100 的主观或模型评分
- `weight_kg`：单件重量，影响物流友好度
- 商品按 `(title, source)` 唯一，重复导入自动更新而非新增

### 接入真实渠道

实现新的采集源即可，打分链路完全复用：

```python
# app/crawler.py
class MySource(Source):
    name = "my-source"

    def fetch(self) -> list[ProductIn]:
        return [ProductIn(title=..., price=..., ...)]

SOURCES[MySource.name] = MySource
```

---

## 项目结构

```
ai-product-selection/
├── app/
│   ├── api.py         # FastAPI 路由
│   ├── config.py      # 环境变量与权重配置
│   ├── crawler.py     # 候选商品采集适配器
│   ├── db.py          # SQLite 存储层
│   ├── llm.py         # OpenAI 兼容大模型接入 + 降级
│   ├── models.py      # Pydantic 数据模型
│   ├── scoring.py     # 规则打分引擎（纯函数，可单测）
│   └── service.py     # 业务编排
├── data/
│   └── sample_products.json
├── scripts/seed_data.py
├── tests/test_scoring.py
├── streamlit_app.py
├── conftest.py
├── requirements.txt
└── .env.example
```

## 测试

```bash
pytest -q
```

## 后续规划

- [ ] 采集器：1688 / 抖音 / 亚马逊榜单
- [ ] 用 LLM 自动估算 `heat`、`virality` 等主观维度，替代手工填写
- [ ] 权重在线调参与 A/B 对比
- [ ] 选品结果导出为采购单 / 上架任务

## License

MIT
