"""打分权重方案：预设、校验与归一化。

权重与「维度」是一一对应的（见 ``app.config.DEFAULT_WEIGHTS``）。
这里不做任何 IO，便于单测与复用。
"""

from __future__ import annotations

from typing import Any, Mapping

from .config import DEFAULT_WEIGHTS, DIMENSION_LABELS

#: 允许出现的维度名
DIMENSIONS: tuple[str, ...] = tuple(DEFAULT_WEIGHTS)

#: 内置预设方案。所有权重之和为 1.0
PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "label": "均衡（默认）",
        "description": "毛利与需求并重，适合大多数类目的通用筛选",
        "weights": dict(DEFAULT_WEIGHTS),
    },
    "margin_first": {
        "label": "毛利优先",
        "description": "把毛利率权重提到 40%，适合预算紧、要求确定性回报的阶段",
        "weights": {
            "margin": 0.40, "demand": 0.18, "competition": 0.14,
            "shipping": 0.08, "repurchase": 0.06, "compliance": 0.06, "virality": 0.08,
        },
    },
    "traffic_first": {
        "label": "流量优先",
        "description": "重需求热度与内容传播力，适合冲量、做爆款测款",
        "weights": {
            "demand": 0.32, "virality": 0.24, "competition": 0.16,
            "margin": 0.16, "shipping": 0.04, "repurchase": 0.04, "compliance": 0.04,
        },
    },
    "low_risk": {
        "label": "低风险",
        "description": "合规与毛利优先，适合新店、怕踩雷的保守打法",
        "weights": {
            "compliance": 0.28, "margin": 0.24, "competition": 0.16,
            "demand": 0.14, "shipping": 0.08, "repurchase": 0.05, "virality": 0.05,
        },
    },
}


def preset_names() -> list[str]:
    """所有预设名。"""
    return list(PRESETS)


def preset_weights(name: str) -> dict[str, float]:
    """取预设权重（返回副本）。"""
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name!r}，可选：{', '.join(PRESETS)}")
    return dict(PRESETS[name]["weights"])


def normalize(weights: Mapping[str, Any]) -> dict[str, float]:
    """清洗并归一化权重。

    - 丢弃未知维度与非法值（负数、非数字）
    - 归一化到总和 1.0；总和为 0 时退回 ``DEFAULT_WEIGHTS``
    - 缺失的维度补 0，保证所有维度都有键
    """
    cleaned: dict[str, float] = {}
    for name in DIMENSIONS:
        try:
            value = float(weights.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        cleaned[name] = max(0.0, value)

    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: round(value / total, 6) for name, value in cleaned.items()}


def validate(weights: Mapping[str, Any]) -> list[str]:
    """校验权重，返回问题列表（空列表表示没问题）。"""
    problems: list[str] = []
    unknown = [name for name in weights if name not in DIMENSIONS]
    if unknown:
        problems.append(f"未知维度：{', '.join(map(str, unknown))}")

    usable = 0
    for name in DIMENSIONS:
        raw = weights.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            problems.append(f"{DIMENSION_LABELS.get(name, name)} 不是数字：{raw!r}")
            continue
        if value < 0:
            problems.append(f"{DIMENSION_LABELS.get(name, name)} 不能为负数：{value}")
            continue
        if value > 0:
            usable += 1

    if usable == 0:
        problems.append("至少要有一个维度的权重大于 0")
    return problems


def parse_weights(text: str) -> dict[str, float]:
    """解析 ``"margin=0.4,demand=0.2"`` 形式的权重串。

    维度名可用英文键或中文标签（如 ``毛利=0.4``）。
    """
    if not text or not text.strip():
        return {}
    label_to_key = {label: key for key, label in DIMENSION_LABELS.items()}
    result: dict[str, float] = {}
    for chunk in text.replace("，", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"权重片段缺少 '='：{chunk!r}")
        name, _, raw = chunk.partition("=")
        name = name.strip()
        key = name if name in DIMENSIONS else label_to_key.get(name)
        if key is None:
            raise ValueError(f"未知维度 {name!r}，可选：{', '.join(DIMENSIONS)} 或其中文名")
        try:
            result[key] = float(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} 的权重不是数字：{raw!r}") from exc
    return result


def merge(*layers: Mapping[str, Any]) -> dict[str, float]:
    """按顺序叠加多层权重（后面的覆盖前面的），结果归一化。"""
    merged: dict[str, float] = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if key in DIMENSIONS:
                merged[key] = value
    return normalize(merged)


def diff(weights_a: Mapping[str, float],
         weights_b: Mapping[str, float]) -> list[dict[str, Any]]:
    """逐维度比较两个权重方案，按变化幅度倒序。"""
    a, b = normalize(weights_a), normalize(weights_b)
    rows = []
    for name in DIMENSIONS:
        left, right = a.get(name, 0.0), b.get(name, 0.0)
        rows.append({
            "dimension": name,
            "label": DIMENSION_LABELS.get(name, name),
            "a": round(left, 4),
            "b": round(right, 4),
            "delta": round(right - left, 4),
        })
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return rows


def describe(weights: Mapping[str, float], top: int = 3) -> str:
    """一句话描述权重重心，例如「毛利 24% > 需求 22% > 竞争 18%」。"""
    normalized = normalize(weights)
    ranked = sorted(normalized.items(), key=lambda pair: pair[1], reverse=True)[:top]
    return " > ".join(
        f"{DIMENSION_LABELS.get(name, name)} {value:.0%}" for name, value in ranked
    )
