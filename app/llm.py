"""大模型接入层：OpenAI 兼容的 Chat Completions 接口。

未配置 key 或调用失败时静默降级为纯规则打分，绝不阻塞主流程。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .config import DIMENSION_LABELS, settings
from .scoring import MAX_LLM_ADJUSTMENT, clamp

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名资深跨境电商选品顾问。
用户会提供商品的基础数据，以及规则引擎给出的各维度得分。
请你输出 JSON（不要包含 markdown 代码块），格式：
{"review": "120 字以内的中文点评，指出最大机会与最大风险，并给出可执行建议",
 "adjustment": 数字，范围 -10 到 10，表示在规则总分基础上的修正值}

要求：点评必须具体，不要复述数据；若无明显理由不要调整分数（adjustment 填 0）。"""


def is_available() -> bool:
    """LLM 是否可用。"""
    return settings.llm_ready


def _build_user_prompt(product, dimensions: dict[str, float], total: float) -> str:
    dims_text = "\n".join(
        f"- {DIMENSION_LABELS.get(key, key)}：{value:.1f}/100" for key, value in dimensions.items()
    )
    return (
        f"商品标题：{product.title}\n"
        f"类目：{product.category}\n"
        f"售价：{product.price:.2f} 元，成本：{product.cost:.2f} 元\n"
        f"重量：{product.weight_kg:g} kg\n"
        f"备注：{product.note or '无'}\n\n"
        f"规则引擎各维度得分：\n{dims_text}\n"
        f"规则总分：{total:.2f}/100\n"
    )


def chat(messages: list[dict[str, str]], temperature: float = 0.3) -> Optional[str]:
    """调用 OpenAI 兼容接口，返回文本内容；失败返回 None。"""
    if not is_available():
        return None
    try:
        import requests
    except ImportError:  # pragma: no cover
        logger.warning("未安装 requests，跳过 LLM 点评")
        return None

    url = f"{settings.llm_base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=settings.llm_timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - 任何异常都不应中断主流程
        logger.warning("LLM 调用失败：%s", exc)
        return None


def extract_json(text: str) -> Any:
    """从模型输出中稳健地提取 JSON 对象或数组。

    容忍 markdown 代码围栏与前后多余文字；解析失败返回 ``None``。
    """
    if not text:
        return None

    cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    for candidate in (cleaned, text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """从模型输出中提取 JSON 对象（非对象时返回 None）。"""
    data = extract_json(text)
    return data if isinstance(data, dict) else None


def review_product(product, dimensions: dict[str, float],
                   total: float) -> tuple[Optional[str], float]:
    """让大模型点评商品并给出有限修正。

    Returns:
        (点评文本或 None, 修正值)。修正值已被限制在 ±MAX_LLM_ADJUSTMENT 内。
    """
    content = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(product, dimensions, total)},
        ]
    )
    if not content:
        return None, 0.0

    data = _extract_json(content)
    if not data:
        # 模型没按格式返回，则把原文当点评，不做修正
        return content.strip()[:500], 0.0

    review = str(data.get("review", "")).strip() or None
    try:
        adjustment = float(data.get("adjustment", 0) or 0)
    except (TypeError, ValueError):
        adjustment = 0.0
    return review, round(clamp(adjustment, -MAX_LLM_ADJUSTMENT, MAX_LLM_ADJUSTMENT), 2)


def apply_adjustment(total: float, adjustment: float) -> float:
    """把修正值应用到总分并截断到 0-100。"""
    return round(clamp(total + adjustment), 2)
