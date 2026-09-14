"""外部数据源适配器。

每个模块实现一个 ``app.crawler.Source`` 子类，把第三方平台的数据转换成
``app.models.ProductIn``，后续打分链路完全复用。
"""

from .douyin import DouyinAPIError, DouyinClient, DouyinSource, to_product

__all__ = ["DouyinAPIError", "DouyinClient", "DouyinSource", "to_product"]
