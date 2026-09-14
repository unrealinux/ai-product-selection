"""数据模型：候选商品与打分明细。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ProductIn(BaseModel):
    """新增/导入商品时的入参。"""

    title: str = Field(..., min_length=1, max_length=200, description="商品标题")
    category: str = Field("未分类", max_length=64, description="类目")
    price: float = Field(0.0, ge=0, description="售价（元）")
    cost: float = Field(0.0, ge=0, description="成本（元）")
    source: str = Field("manual", max_length=32, description="来源渠道")
    url: str = Field("", max_length=500, description="商品链接")
    heat: float = Field(50.0, ge=0, le=100, description="需求热度 0-100")
    competition: float = Field(50.0, ge=0, le=100, description="竞争度 0-100，越低越好")
    weight_kg: float = Field(0.5, ge=0, le=50, description="单件重量（kg）")
    repurchase: float = Field(50.0, ge=0, le=100, description="复购潜力 0-100")
    compliance_risk: float = Field(20.0, ge=0, le=100, description="合规风险 0-100，越低越好")
    virality: float = Field(50.0, ge=0, le=100, description="内容传播力 0-100")
    note: str = Field("", max_length=1000, description="备注")

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title 不能为空")
        return value


class Product(ProductIn):
    """带主键与时间戳的完整商品记录。"""

    id: int
    created_at: datetime


class ScoreBreakdown(BaseModel):
    """单次打分的明细。"""

    product_id: int
    total: float = Field(..., ge=0, le=100, description="加权总分 0-100")
    grade: str = Field(..., description="等级 S/A/B/C/D")
    dimensions: dict[str, float] = Field(default_factory=dict, description="各维度得分 0-100")
    profit_margin: float = Field(0.0, description="毛利率 0-1")
    advice: str = Field("", description="规则引擎给出的选品建议")
    llm_review: Optional[str] = Field(None, description="大模型点评，未启用时为 null")
    llm_adjustment: float = Field(0.0, description="大模型给出的分数修正值")
    scored_at: datetime = Field(default_factory=datetime.now)


class Stats(BaseModel):
    """看板统计信息。"""

    total: int
    scored: int
    avg_score: float
    grade_distribution: dict[str, int]
    top_categories: list[dict[str, Any]]
