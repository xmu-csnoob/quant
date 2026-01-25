"""
数据获取器基础模块

包含：
- Exchange: 交易所枚举
- BaseDataFetcher: 抽象基类
- 异常类定义
"""

from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd


class Exchange(Enum):
    """证券交易所枚举"""

    SSE = "SSE"  # 上海证券交易所
    SZSE = "SZSE"  # 深圳证券交易所
    BSE = "BSE"  # 北京证券交易所

    def __str__(self) -> str:
        return self.value


class DataFetcherError(Exception):
    """数据获取器基础异常"""

    pass


class TokenNotFoundError(DataFetcherError):
    """API Token 未找到异常"""

    pass


class DataFetchError(DataFetcherError):
    """数据获取失败异常"""

    pass


class DataValidationError(DataFetcherError):
    """数据验证失败异常"""

    pass


class BaseDataFetcher(ABC):
    """
    数据获取器抽象基类

    定义统一的接口，支持多种数据源实现：
    - MockDataFetcher: 模拟数据（开发/测试）
    - TushareDataFetcher: Tushare 真实数据
    - AkShareDataFetcher: AkShare 备用数据源
    """

    @abstractmethod
    def get_stock_list(self, exchange: Exchange) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            exchange: 交易所枚举

        Returns:
            股票列表 DataFrame，包含列：
            - ts_code: 股票代码 (如 600000.SH)
            - symbol: 股票代码（数字部分）
            - name: 股票名称
            - area: 地域
            - industry: 行业
            - list_date: 上市日期 (YYYYMMDD)

        Raises:
            DataFetchError: 数据获取失败
        """
        pass

    @abstractmethod
    def get_daily_price(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取日线行情数据

        Args:
            ts_code: 股票代码（如 600000.SH）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            日线数据 DataFrame，包含列：
            - ts_code: 股票代码
            - trade_date: 交易日期（YYYYMMDD）
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量（手）
            - amount: 成交额（千元）

        Raises:
            DataFetchError: 数据获取失败
        """
        pass

    @abstractmethod
    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线数据

        Args:
            index_code: 指数代码（如 000001.SH - 上证综指）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            指数数据 DataFrame，列结构同日线数据

        Raises:
            DataFetchError: 数据获取失败
        """
        pass
