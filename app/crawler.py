"""候选商品采集适配器。

当前内置：
- ``sample``：读取 ``data/sample_products.json`` 示例数据（开箱即用）
- ``json``  ：读取任意本地 JSON / JSONL 文件

真实渠道（1688、抖音、亚马逊等）请实现新的 ``Source`` 子类并注册到 ``SOURCES``，
保持 ``fetch()`` 返回 ``list[ProductIn]`` 的约定即可，后续打分链路无需改动。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import BASE_DIR
from .models import ProductIn

SAMPLE_PATH = BASE_DIR / "data" / "sample_products.json"


class Source:
    """采集源基类。"""

    name = "base"

    def fetch(self) -> list[ProductIn]:  # pragma: no cover - 抽象方法
        raise NotImplementedError


class SampleSource(Source):
    """内置示例数据源。"""

    name = "sample"

    def __init__(self, path: Path | str = SAMPLE_PATH) -> None:
        self.path = Path(path)

    def fetch(self) -> list[ProductIn]:
        return load_json(self.path)


class JsonFileSource(Source):
    """任意本地 JSON / JSONL 文件数据源。"""

    name = "json"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def fetch(self) -> list[ProductIn]:
        return load_json(self.path)


def load_json(path: Path | str) -> list[ProductIn]:
    """读取 JSON 数组或 JSONL 文件，统一解析为 ProductIn 列表。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[dict]
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    products: list[ProductIn] = []
    for index, record in enumerate(records, start=1):
        record.setdefault("source", path.stem)
        try:
            products.append(ProductIn(**record))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path.name} 第 {index} 条数据校验失败：{exc}") from exc
    return products


SOURCES: dict[str, type[Source]] = {
    SampleSource.name: SampleSource,
    JsonFileSource.name: JsonFileSource,
}

#: 需要第三方依赖或凭据、延迟导入的数据源（名字 → 获取实例的工厂）
LAZY_SOURCES = ("douyin",)


def _build_douyin(path: Path | str | None, options: dict):
    """构造抖音数据源（延迟导入，未配置凭据时也能正常 import 本项目）。"""
    from .sources.douyin import DouyinSource

    options = dict(options)
    if path and "keywords" not in options:
        options["keywords"] = [item.strip() for item in str(path).split(",") if item.strip()]
    return DouyinSource(**options)


def get_source(name: str, path: Path | str | None = None,
               options: dict | None = None) -> Source:
    """按名字获取采集源。

    Args:
        name: 数据源名字，可选 ``sample`` / ``json`` / ``douyin``。
        path: ``json`` 源的文件路径；``douyin`` 源可传逗号分隔的关键词。
        options: 数据源构造参数，例如 ``{"page_size": 20, "max_pages": 2}``。
    """
    if name not in SOURCES and name not in LAZY_SOURCES:
        available = ", ".join([*SOURCES, *LAZY_SOURCES])
        raise KeyError(f"未知数据源 {name!r}，可选：{available}")

    if name in LAZY_SOURCES:
        return _build_douyin(path, options or {})

    cls = SOURCES[name]
    if cls is JsonFileSource:
        if path is None:
            raise ValueError("json 数据源必须提供 path")
        return cls(path)
    return cls(path) if path else cls()


def fetch_all(sources: Iterable[Source]) -> list[ProductIn]:
    """聚合多个数据源的结果。"""
    items: list[ProductIn] = []
    for source in sources:
        items.extend(source.fetch())
    return items
