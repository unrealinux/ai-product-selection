"""主图相似度（dHash）测试。

盯住四件事：

1. **归一化**：``//img...``、裸域名、占位值「无」都要处理对，否则会去下载垃圾
2. **算不出来 ≠ 不相似**：缺图 / 下载失败 / 解码失败必须返回 ``None``，不能返回 0
3. **缓存**：同一 URL 只下载一次；缓存损坏不影响主流程
4. **阈值语义**：相同图 → 1.0，完全互补 → 接近 0
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app.imagehash import (
    ImageHashCache,
    dhash_from_bytes,
    dhash_from_image,
    fetch_bytes,
    hamming_hex,
    image_similarity,
    is_available,
    normalize_image_url,
    similarity_from_hashes,
)


# --------------------------------------------------------------------------- #
# 造图
# --------------------------------------------------------------------------- #

def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def horizontal_gradient(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """左黑右白：相邻像素「左暗于右」，dHash 每一位都是 0。"""
    image = Image.new("L", size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = int(255 * x / size[0])
    return image


def left_bright(size: tuple[int, int] = (64, 64)) -> Image.Image:
    """左白右黑：与 horizontal_gradient 正好互补，dHash 全为 1。"""
    image = Image.new("L", size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = int(255 * (size[0] - 1 - x) / size[0])
    return image


def solid(level: int = 128, size: tuple[int, int] = (64, 64)) -> Image.Image:
    return Image.new("L", size, color=level)


# --------------------------------------------------------------------------- #
# URL 归一化
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expected", [
    (None, ""),
    ("", ""),
    ("   ", ""),
    ("无", ""),
    ("-", ""),
    ("0", ""),
    ("//img.alicdn.com/bao/a.jpg", "https://img.alicdn.com/bao/a.jpg"),
    ("https://img.alicdn.com/a.jpg", "https://img.alicdn.com/a.jpg"),
    ("http://img.alicdn.com/a.jpg", "http://img.alicdn.com/a.jpg"),
    ("img.alicdn.com/a.jpg", "https://img.alicdn.com/a.jpg"),
    ("  https://x.com/a.jpg  ", "https://x.com/a.jpg"),
])
def test_normalize_image_url(raw, expected):
    assert normalize_image_url(raw) == expected


def test_normalize_image_url_rejects_non_urls():
    assert normalize_image_url("随便一句话") == ""
    assert normalize_image_url("D:/images/a.jpg") == ""


def test_normalize_image_url_truncates():
    long_url = "https://img.alicdn.com/" + "a" * 600
    assert len(normalize_image_url(long_url)) == 500


# --------------------------------------------------------------------------- #
# dHash
# --------------------------------------------------------------------------- #

def test_pillow_is_available_in_this_environment():
    assert is_available() is True


def test_dhash_is_deterministic():
    data = png_bytes(horizontal_gradient())
    assert dhash_from_bytes(data) == dhash_from_bytes(data)
    assert len(dhash_from_bytes(data)) == 16  # 8x8 bit → 16 位十六进制


def test_dhash_of_uniform_image_is_all_zero():
    """纯色图没有任何「左>右」的关系，哈希应为全 0。"""
    assert dhash_from_bytes(png_bytes(solid())) == "0" * 16


def test_dhash_distinguishes_different_images():
    a = dhash_from_bytes(png_bytes(horizontal_gradient()))
    b = dhash_from_bytes(png_bytes(left_bright()))
    assert a != b
    assert similarity_from_hashes(a, b) < 0.9


def test_dhash_returns_none_on_broken_bytes():
    assert dhash_from_bytes("这不是图片".encode("utf-8")) is None
    assert dhash_from_bytes(b"") is None


def test_dhash_from_image_accepts_pil_image():
    assert dhash_from_image(horizontal_gradient()) == dhash_from_bytes(
        png_bytes(horizontal_gradient())
    )


# --------------------------------------------------------------------------- #
# 距离与相似度
# --------------------------------------------------------------------------- #

def test_hamming_and_similarity():
    assert hamming_hex("0" * 16, "0" * 16) == 0
    assert hamming_hex("0" * 16, "f" * 16) == 64
    assert similarity_from_hashes("0" * 16, "0" * 16) == 1.0
    assert similarity_from_hashes("0" * 16, "f" * 16) == 0.0


def test_hamming_and_similarity_reject_bad_input():
    assert hamming_hex("abc", "abcdef") > 64
    assert hamming_hex("zz", "zz") > 64
    assert similarity_from_hashes("abc", "abcdef") is None
    assert similarity_from_hashes("", "") is None


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #

def test_hash_cache_round_trip(tmp_path):
    path = tmp_path / "hashes.json"
    cache = ImageHashCache(path)
    cache.put("https://x.com/a.jpg", "abcd1234abcd1234")
    cache.save()

    assert ImageHashCache(path).get("https://x.com/a.jpg") == "abcd1234abcd1234"


def test_hash_cache_disabled_is_noop(tmp_path):
    cache = ImageHashCache(tmp_path / "h.json", enabled=False)
    cache.put("u", "d")
    cache.save()
    assert cache.get("u") is None
    assert not (tmp_path / "h.json").exists()


def test_hash_cache_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ 不是合法 JSON", encoding="utf-8")
    assert ImageHashCache(path).get("u") is None


# --------------------------------------------------------------------------- #
# 端到端（注入假下载器，不碰网络）
# --------------------------------------------------------------------------- #

class FakeFetcher:
    def __init__(self, mapping: dict[str, bytes | None]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes | None:
        self.calls.append(url)
        return self.mapping.get(url)


def test_identical_images_score_one(tmp_path):
    data = png_bytes(horizontal_gradient())
    fetcher = FakeFetcher({"https://x.com/a.jpg": data, "https://x.com/b.jpg": data})

    score = image_similarity("https://x.com/a.jpg", "https://x.com/b.jpg",
                             cache=ImageHashCache(tmp_path / "h.json"), fetcher=fetcher)

    assert score == pytest.approx(1.0)


def test_different_images_score_low(tmp_path):
    fetcher = FakeFetcher({
        "https://x.com/a.jpg": png_bytes(horizontal_gradient()),
        "https://x.com/b.jpg": png_bytes(left_bright()),
    })
    score = image_similarity("https://x.com/a.jpg", "https://x.com/b.jpg",
                             cache=ImageHashCache(tmp_path / "h.json"), fetcher=fetcher)
    assert score is not None and score < 0.9


def test_missing_url_returns_none_without_downloading(tmp_path):
    fetcher = FakeFetcher({})
    assert image_similarity("", "https://x.com/b.jpg",
                            cache=ImageHashCache(tmp_path / "h.json"),
                            fetcher=fetcher) is None
    assert image_similarity("无", "https://x.com/b.jpg",
                            cache=ImageHashCache(tmp_path / "h.json"),
                            fetcher=fetcher) is None
    assert fetcher.calls == []


def test_download_failure_returns_none_not_zero(tmp_path):
    """关键：下载失败必须返回 None，调用方不能把它当成「不相似」。"""
    fetcher = FakeFetcher({"https://x.com/a.jpg": None})
    assert image_similarity("https://x.com/a.jpg", "https://x.com/b.jpg",
                            cache=ImageHashCache(tmp_path / "h.json"),
                            fetcher=fetcher) is None


def test_broken_image_returns_none(tmp_path):
    fetcher = FakeFetcher({
        "https://x.com/a.jpg": b"not an image",
        "https://x.com/b.jpg": png_bytes(solid()),
    })
    assert image_similarity("https://x.com/a.jpg", "https://x.com/b.jpg",
                            cache=ImageHashCache(tmp_path / "h.json"),
                            fetcher=fetcher) is None


def test_hash_cache_avoids_second_download(tmp_path):
    data = png_bytes(horizontal_gradient())
    fetcher = FakeFetcher({"https://x.com/a.jpg": data, "https://x.com/b.jpg": data})
    cache = ImageHashCache(tmp_path / "h.json")

    image_similarity("https://x.com/a.jpg", "https://x.com/b.jpg", cache=cache, fetcher=fetcher)
    image_similarity("https://x.com/a.jpg", "https://x.com/b.jpg", cache=cache, fetcher=fetcher)

    assert fetcher.calls.count("https://x.com/a.jpg") == 1
    assert fetcher.calls.count("https://x.com/b.jpg") == 1


def test_fetch_bytes_never_raises_on_bad_url():
    assert fetch_bytes("https://127.0.0.1:1/nope.jpg", timeout=1) is None
