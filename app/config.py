"""全局配置：全部通过环境变量注入，便于本地与容器化部署。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv 为可选依赖
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


#: 打分权重，权重之和必须为 1.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "demand": 0.22,       # 需求热度
    "competition": 0.18,  # 竞争度（反向指标）
    "margin": 0.24,       # 毛利率
    "shipping": 0.08,     # 物流友好度（重量）
    "repurchase": 0.08,   # 复购潜力
    "compliance": 0.08,   # 合规风险（反向指标）
    "virality": 0.12,     # 内容传播力
}

#: 各维度中文名，前端展示用
DIMENSION_LABELS: dict[str, str] = {
    "demand": "需求热度",
    "competition": "竞争度",
    "margin": "毛利率",
    "shipping": "物流友好",
    "repurchase": "复购潜力",
    "compliance": "合规安全",
    "virality": "传播潜力",
}


@dataclass(frozen=True)
class Settings:
    """运行时配置。"""

    base_dir: Path = BASE_DIR
    db_path: Path = field(
        default_factory=lambda: (BASE_DIR / os.getenv("APS_DB_PATH", "data/products.db"))
    )

    llm_enabled: bool = field(default_factory=lambda: _env_bool("APS_LLM_ENABLED"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("APS_LLM_BASE_URL", "").rstrip("/")
    )
    llm_api_key: str = field(default_factory=lambda: os.getenv("APS_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("APS_LLM_MODEL", "gpt-4o-mini"))
    llm_timeout: int = field(default_factory=lambda: _env_int("APS_LLM_TIMEOUT", 60))

    api_host: str = field(default_factory=lambda: os.getenv("APS_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("APS_API_PORT", 8000))

    # ---- 抖音开放平台（抖店 / 精选联盟）----
    # 文档：https://op.jinritemai.com/docs/guide-docs/148/814
    douyin_base_url: str = field(
        default_factory=lambda: os.getenv(
            "APS_DOUYIN_BASE_URL", "https://openapi-fxg.jinritemai.com"
        ).rstrip("/")
    )
    douyin_app_key: str = field(default_factory=lambda: os.getenv("APS_DOUYIN_APP_KEY", ""))
    douyin_app_secret: str = field(
        default_factory=lambda: os.getenv("APS_DOUYIN_APP_SECRET", "")
    )
    douyin_access_token: str = field(
        default_factory=lambda: os.getenv("APS_DOUYIN_ACCESS_TOKEN", "")
    )
    douyin_shop_id: str = field(default_factory=lambda: os.getenv("APS_DOUYIN_SHOP_ID", ""))
    douyin_sign_method: str = field(
        default_factory=lambda: os.getenv("APS_DOUYIN_SIGN_METHOD", "hmac-sha256").lower()
    )
    douyin_timeout: int = field(default_factory=lambda: _env_int("APS_DOUYIN_TIMEOUT", 30))

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @property
    def llm_ready(self) -> bool:
        """LLM 是否具备可用条件。"""
        return bool(self.llm_enabled and self.llm_base_url and self.llm_api_key)

    @property
    def douyin_ready(self) -> bool:
        """抖音数据源是否具备可用条件。"""
        return bool(
            self.douyin_app_key
            and self.douyin_app_secret
            and self.douyin_access_token
        )


settings = Settings()
