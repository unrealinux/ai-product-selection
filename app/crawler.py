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


def get_source(name: str, path: Path | str | None = None) -> Source:
    """按名字获取采集源。"""
    if name not in SOURCES:
        raise KeyError(f"未知数据源 {name!r}，可选：{', '.join(SOURCES)}")
    cls = SOURCES[name]
    if cls is JsonFileSource:
        if path is None:
            raise ValueError("json 数据源必须提供 --path")
        return cls(path)
    return cls(path) if path else cls()


def fetch_all(sources: Iterable[Source]) -> list[ProductIn]:
    """聚合多个数据源的结果。"""
    items: list[ProductIn] = []
    for source in sources:
        items.extend(source.fetch())
    return items
