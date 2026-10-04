"""pytest 根配置：把项目根目录加入 sys.path，使 `import app` 可用。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把 ``app.db`` 的默认库指向临时文件，避免测试污染真实的 ``data/products.db``。

    ``Settings`` 是 frozen dataclass，字段改不动；但 ``db.connect()`` 每次调用都读
    ``db.settings.db_path``，所以替换模块级的 ``settings`` 就足够。service / api
    层不接收 ``db_path`` 参数，它们最终也都走 ``app.db`` 的这个默认值。
    """
    from app import db

    path = tmp_path / "products.db"
    monkeypatch.setattr(db, "settings", SimpleNamespace(db_path=path))
    db.init_db(path)
    return path
