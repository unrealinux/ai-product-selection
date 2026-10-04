"""商品主图相似度：dHash 感知哈希 + URL 缓存。

为什么需要它
------------
成本对齐只能靠标题，而标题相似度对「图片明显不同的两款货」是盲的 ——
同一个标题措辞可以对应外形完全不同的商品。主图是最直接的证据：

* 淘宝 ``item-detail`` / ``item-search`` 都返回 ``picPath``（已实测）
* 1688 / 分销后台导出的商品表通常带「主图链接」列

两边都有图时，用 **dHash（差分感知哈希）** 比较即可：它只关心相邻像素的
明暗关系，因此缩略图、加水印、轻微裁剪都还能对上，而不同商品通常差异很大。

.. warning::
   图片相似度是**证据之一，不是判定**。同款不同色、同款不同角度可能被压到较低分，
   因此阈值给得保守（见 ``costlink.IMAGE_WEAK`` / ``IMAGE_STRONG``），
   且**只用于复核灰区候选**，不参与高置信匹配。

Pillow 为可选依赖：未安装时 :func:`is_available` 返回 False，
所有函数返回 ``None``，调用方应保持原有匹配结果不变。
"""

from __future__ import annotations

import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

from .config import settings

logger = logging.getLogger(__name__)

#: dHash 边长。8 → 64 bit → 16 位十六进制
DEFAULT_HASH_SIZE = 8

#: 占位值，视为「没有图」
_EMPTY_TOKENS = {"", "-", "--", "/", "无", "none", "null", "n/a", "na", "nan", "0"}

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def normalize_image_url(raw: Any) -> str:
    """把各种写法的图片地址统一成可用 URL。

    处理：去空白、协议相对地址（``//img...`` → ``https://img...``）、
    缺协议的裸域名、以及「无 / - / 0」这类占位值。
    """
    if raw is None:
        return ""
    text = str(raw).strip().strip('"').strip("'")
    if text.lower() in _EMPTY_TOKENS:
        return ""
    if text.startswith("//"):
        return f"https://{text[2:]}"
    if _SCHEME.match(text):
        return text[:500]
    # 裸域名（1688 导出表里常见 img.alicdn.com/... 这种）
    if re.match(r"^[\w.\-]+\.[a-z]{2,}/", text, re.IGNORECASE):
        return f"https://{text}"[:500]
    return ""


def is_available() -> bool:
    """Pillow 是否可用（未安装时整个图片链路自动跳过）。"""
    try:
        import PIL  # noqa: F401
    except ImportError:  # pragma: no cover - 取决于运行环境
        return False
    return True


# --------------------------------------------------------------------------- #
# dHash
# --------------------------------------------------------------------------- #

def dhash_from_image(image: Any, size: int = DEFAULT_HASH_SIZE) -> Optional[str]:
    """对 PIL Image 求 dHash，返回定长十六进制字符串。"""
    try:
        from PIL import Image

        gray = image.convert("L").resize((size + 1, size), Image.LANCZOS)
        # Pillow 12 弃用了 getdata()，但它要更晚才移除，这里两者都兼容
        if hasattr(gray, "get_flattened_data"):
            pixels = list(gray.get_flattened_data())
        else:  # pragma: no cover - 取决于 Pillow 版本
            pixels = list(gray.getdata())
    except Exception as exc:  # noqa: BLE001 - 图片损坏不应中断流程
        logger.warning("计算 dHash 失败：%s", exc)
        return None

    bits = 0
    width = size + 1
    for row in range(size):
        offset = row * width
        for col in range(size):
            # 左像素比右像素亮 → 1，否则 0
            bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    return f"{bits:0{size * size // 4}x}"


def dhash_from_bytes(data: bytes, size: int = DEFAULT_HASH_SIZE) -> Optional[str]:
    """对图片字节求 dHash。无法解码时返回 ``None``。"""
    if not data:
        return None
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            return dhash_from_image(image, size=size)
    except Exception as exc:  # noqa: BLE001
        logger.warning("图片解码失败，跳过：%s", exc)
        return None


def hamming_hex(a: str, b: str) -> int:
    """两个十六进制哈希的汉明距离。长度不同时返回一个很大的值。"""
    if not a or not b or len(a) != len(b):
        return 1 << 30
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 1 << 30


def similarity_from_hashes(a: str, b: str) -> Optional[float]:
    """由两个哈希算相似度 0-1（1 = 完全一致）。"""
    if not a or not b or len(a) != len(b):
        return None
    bits = len(a) * 4
    distance = hamming_hex(a, b)
    if distance > bits:
        return None
    return round(1.0 - distance / bits, 6)


# --------------------------------------------------------------------------- #
# 下载与缓存
# --------------------------------------------------------------------------- #

def fetch_bytes(url: str, timeout: Optional[int] = None) -> Optional[bytes]:
    """下载图片字节。失败返回 ``None``，不抛异常。"""
    try:
        import requests
    except ImportError:  # pragma: no cover - requests 已是必需依赖
        logger.warning("未安装 requests，无法下载图片")
        return None
    try:
        response = requests.get(
            url,
            timeout=timeout or settings.image_timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ai-product-selection/1.0)"},
        )
        if response.status_code != 200:
            logger.warning("图片下载失败 HTTP %s：%s", response.status_code, url)
            return None
        return response.content
    except Exception as exc:  # noqa: BLE001 - 网络问题不应中断匹配
        logger.warning("图片下载异常：%s", exc)
        return None


class ImageHashCache:
    """按图片 URL 缓存 dHash 的 JSON 文件，避免重复下载。"""

    def __init__(self, path: Path | str | None = None, *, enabled: bool = True) -> None:
        self.path = Path(path) if path else settings.image_cache_path
        self.enabled = enabled
        self._data: dict[str, str] = {}
        self._dirty = False
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._data = {str(k): str(v) for k, v in payload.items() if v}
        except Exception as exc:  # noqa: BLE001 - 缓存损坏不应影响主流程
            logger.warning("图片哈希缓存读取失败，将忽略：%s", exc)
            self._data = {}

    def get(self, url: str) -> Optional[str]:
        if not self.enabled:
            return None
        return self._data.get(url)

    def put(self, url: str, digest: str) -> None:
        if not self.enabled or not digest:
            return
        self._data[url] = digest
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._dirty = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("图片哈希缓存写入失败：%s", exc)

    def __len__(self) -> int:
        return len(self._data)


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #

def image_similarity(
    url_a: Any,
    url_b: Any,
    *,
    cache: Optional[ImageHashCache] = None,
    fetcher: Optional[Callable[[str], Optional[bytes]]] = None,
    size: int = DEFAULT_HASH_SIZE,
) -> Optional[float]:
    """两张主图的相似度 0-1；任一环节拿不到就返回 ``None``。

    ``None`` 表示「算不出来」，绝不能当成「不相似（0）」—— 调用方必须区别对待，
    否则缺少图片的商品会被静默判为不同款。

    Args:
        fetcher: 注入用（测试里传假下载器）。默认走 :func:`fetch_bytes`。
    """
    a = normalize_image_url(url_a)
    b = normalize_image_url(url_b)
    if not a or not b:
        return None
    if not is_available():
        logger.info("未安装 Pillow，跳过图片相似度")
        return None

    cache = cache if cache is not None else ImageHashCache()
    fetcher = fetcher or fetch_bytes

    digests: list[str] = []
    for url in (a, b):
        digest = cache.get(url)
        if digest is None:
            data = fetcher(url)
            digest = dhash_from_bytes(data, size=size) if data else None
            if digest is None:
                return None
            cache.put(url, digest)
        digests.append(digest)

    cache.save()
    return similarity_from_hashes(digests[0], digests[1])
