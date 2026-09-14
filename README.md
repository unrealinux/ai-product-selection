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

数据来源既支持本地 JSON，也内置了**抖音精选联盟官方 API** 采集器（见 [接入抖音](#接入抖音精选联盟官方-api)）。

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

看板包含五个页签：**选品榜单 / 数据导入 / 抖音拉取 / 商品录入 / 数据概览**，支持 CSV 导出。

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

## 接入抖音（精选联盟官方 API）

内置 `douyin` 数据源，直接调用抖店开放平台的选品广场商品搜索接口
`buyin.kolMaterialsProductsSearch`，走**官方 API**而非页面解析。

- API 调用指南与签名算法：<https://op.jinritemai.com/docs/guide-docs/148/814>
- 选品商品搜索接口文档：<https://op.jinritemai.com/docs/api-docs/61/1725>

### 1. 申请应用

1. 到 [抖店开放平台](https://op.jinritemai.com/) 注册并创建应用（自用型即可）
2. 取得 `app_key` / `app_secret`，并在应用里开通精选联盟相关权限
3. 记录你的 `shop_id`

### 2. 配置并换取 access_token

```dotenv
APS_DOUYIN_APP_KEY=你的app_key
APS_DOUYIN_APP_SECRET=你的app_secret
APS_DOUYIN_SHOP_ID=你的shop_id
```

```bash
python scripts/douyin_fetch.py --check    # 检查配置，不发请求
python scripts/douyin_fetch.py --token    # 自用型应用换取 access_token
# 把返回的 access_token 写入 .env 的 APS_DOUYIN_ACCESS_TOKEN
```

### 3. 拉取选品

```bash
# 试拉取，只看结果不入库
python scripts/douyin_fetch.py --keywords "咖啡,保温杯" --pages 2 --dry-run

# 正式拉取 + 导入 + 打分
python scripts/douyin_fetch.py --keywords "咖啡,保温杯" --pages 2 --score

# 只要佣金率 ≥ 10% 的商品（乘 100，即 1000）
python scripts/douyin_fetch.py --keywords "咖啡" --cos-ratio-min 1000 --score
```

或直接走 HTTP 接口：

```bash
curl -X POST http://127.0.0.1:8000/import -H "Content-Type: application/json" -d '{
  "source": "douyin",
  "path": "咖啡,保温杯",
  "options": {"page_size": 20, "max_pages": 2, "search_type": 1},
  "score": true,
  "use_llm": false
}'
```

### 4. 在 Streamlit 里拉取

看板的 **🎯 抖音拉取** 页签提供图形化入口：

- 顶部直接显示配置状态（app_key / shop_id / 签名算法）
- 参数表单：关键词、翻页数、每页条数、召回排序、排序方向、最低佣金率、类目 ID、仅保留在售
- 结果预览表（售价 / 商家实收 / 佣金率 / 各维度得分）+ CSV 导出
- 一键导入并打分，完成后榜单页同步刷新
- 未配置凭据时，该页签直接展示下面三步的配置指引，不会报错

接口错误会翻译成可执行的下一步：access_token 失效 → 提示重新执行 `--token`；
限流（code 9）→ 提示减少翻页；签名失败（code 11）→ 提示核对 `APS_DOUYIN_SIGN_METHOD`。

### 数据字段来自哪里（重要）

选品接口只提供部分字段，**哪些维度是真实数据、哪些是缺省值**必须说清楚：

| 维度 | 来源 | 说明 |
| --- | --- | --- |
| 需求热度 | ✅ 接口 | `sales`（历史总销量）取对数映射，100 万销量 = 100 分 |
| 竞争度 | ✅ 接口 | 同条件下 `total`（在售商品数）取对数映射 |
| 毛利率 | ✅ 接口 | `kol_cos_fee` / `kol_cos_ratio`（达人佣金）换算 |
| 内容传播力 | ⚠️ 代理 | 用 `kol_cos_ratio` 代理（佣金越高越易撬动达人内容） |
| 重量 | ❌ 缺省 0.5 | 接口未返回，需人工复核 |
| 复购潜力 | ❌ 缺省 50 | 接口未返回，需人工复核 |
| 合规风险 | ❌ 缺省 20 | 接口未返回，需人工复核 |

**因此抖音来源的商品有 3 个维度是常数**，总分主要由销量与佣金率驱动。这是当前实现的
已知局限：`note` 字段会写明「需人工复核」，`app.sources.douyin.FIELD_PROVENANCE`
也把每个维度的来源写进了代码，可用 `scripts/douyin_fetch.py --check` 打印出来。

三条与官方口径相关的换算约定（代码已处理）：

- `price`、`kol_cos_fee`、`coupon_price` 单位是**分**，已换算成元
- `kol_cos_ratio` 是**百分数乘 100**（`10.00` 表示 10%），已除以 100
- `cost` 被反推为「售价 − 达人佣金」，因此算出的毛利率**等于达人佣金率** —— 这是
  **分销带货视角**。自营商家请自行覆盖 `cost`

### 签名实现

严格按官方文档第六节实现，并用**官方样例字符串**做了逐字断言
（`tests/test_douyin.py::test_sign_string_matches_official_doc_example`）：

1. `param_json` 键按字母升序、分隔符无空格；转义 `&`、`<`、`>` 与退格符
2. 全部请求参数按字母排序，其中 **`access_token` 与 `sign_method` 不参与加密**
3. 拼接为 `key1value1key2value2...`
4. 把 **`app_secret` 拼在字符串两端**
5. 对结果做 HMAC-SHA256（密钥同为 `app_secret`）或 MD5，取小写十六进制

> 几个容易踩的坑，代码里都已处理：`param_json` 走请求 body 而其余公共参数走 query；
> 成功码是 `10000` 而非 `0`；`token.create` 不携带 `access_token`；
> 官方标记 `sign_method` 默认 `md5` 但推荐迁到 `hmac-sha256`，本项目默认用后者。

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
| `GET` | `/douyin/status` | 抖音数据源配置状态 + 官返回码释义 |
| `POST` | `/douyin/token` | 换取 access_token（调试用，结果不落盘） |

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

内置的 `douyin` 数据源就在 `app/sources/douyin.py`，可直接当作模板：实现 `fetch()`
返回 `list[ProductIn]`，打分链路完全复用。

```python
# app/sources/my_platform.py
from ..crawler import Source
from ..models import ProductIn

class MySource(Source):
    name = "my-platform"

    def __init__(self, keywords=None, **options):
        ...

    def fetch(self) -> list[ProductIn]:
        return [ProductIn(title=..., price=..., ...)]
```

然后在 `app/crawler.py` 的 `LAZY_SOURCES` 里登记名字，即可用统一入口调用：

```python
source = crawler.get_source("my-platform", "关键词1,关键词2", {"page_size": 20})
products = source.fetch()
```

---

## 项目结构

```
ai-product-selection/
├── app/
│   ├── api.py         # FastAPI 路由
│   ├── config.py      # 环境变量与权重配置
│   ├── crawler.py     # 采集源注册表（延迟加载第三方源）
│   ├── db.py          # SQLite 存储层
│   ├── llm.py         # OpenAI 兼容大模型接入 + 降级
│   ├── models.py      # Pydantic 数据模型
│   ├── scoring.py     # 规则打分引擎（纯函数，可单测）
│   ├── service.py     # 业务编排
│   └── sources/
│       └── douyin.py  # 抖音精选联盟官方 API 客户端 + 字段映射
├── data/
│   └── sample_products.json
├── scripts/
│   ├── douyin_fetch.py
│   └── seed_data.py
├── tests/
│   ├── test_douyin.py
│   └── test_scoring.py
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

- [x] 采集器：抖音精选联盟（官方 API，`buyin.kolMaterialsProductsSearch`）
- [ ] 采集器：1688 / 亚马逊榜单
- [ ] 补齐抖音商品的重量 / 复购 / 合规维度（接入商品详情 API 或人工补录）
- [ ] 用 LLM 自动估算 `heat`、`virality` 等主观维度，替代手工填写
- [ ] 权重在线调参与 A/B 对比
- [ ] 选品结果导出为采购单 / 上架任务

## License

MIT
