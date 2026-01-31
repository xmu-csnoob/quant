"""
AkShare 数据获取器

从 AkShare 获取 A 股市场数据

优点：
1. 免费，无需 Token
2. 无频率限制
3. 数据来源于公开网站（东方财富、新浪等）
4. 支持多种数据类型

注意：
- 数据依赖第三方网站稳定性
- 可能有反爬限制（但 AkShare 已处理）
"""

import akshare as ak
import pandas as pd
import time
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from loguru import logger

from src.data.fetchers.base import (
    BaseDataFetcher,
    Exchange,
    DataFetchError
)


class AkShareDataFetcher(BaseDataFetcher):
    """
    AkShare 数据获取器

    特点：
    1. 完全免费，无需申请 Token
    2. 无调用频率限制
    3. 数据源稳定（东方财富、新浪等）
    4. 支持全市场数据

    用途：
    - 开发阶段数据获取
    - Tushare 的备用数据源
    - 需要高频调用的场景

    文档：
    https://akshare.akfamily.xyz/
    """

    # 交易所代码映射
    EXCHANGE_MAP = {
        Exchange.SSE: "sh",
        Exchange.SZSE: "sz",
    }

    # 添加延迟避免反爬（秒）
    REQUEST_DELAY = 0.5

    def __init__(self, delay: float = 0.5):
        """
        初始化 AkShare 数据获取器

        Args:
            delay: 请求间隔（秒），避免被反爬
        """
        self.delay = delay
        logger.info("AkShareDataFetcher initialized successfully")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.ConnectionError))
    )
    def get_stock_list(self, exchange: Exchange) -> pd.DataFrame:
        """
        获取股票列表

        使用 stock_info_a_code_name API，更稳定

        Args:
            exchange: 交易所枚举

        Returns:
            股票列表 DataFrame，包含列：
            - ts_code: 股票代码
            - symbol: 股票代码（数字部分）
            - name: 股票名称
            - area: 地域
            - industry: 行业
            - market: 市场
            - list_date: 上市日期

        Raises:
            DataFetchError: 数据获取失败
        """
        try:
            logger.info(f"Fetching stock list from {exchange.value} (AkShare)")

            # 使用更稳定的 API - 获取股票信息代码名称
            df_info = ak.stock_info_a_code_name()

            if df_info.empty:
                logger.warning(f"No stocks found from AkShare")
                return pd.DataFrame()

            # df_info 列: code, name
            # 筛选指定交易所
            exchange_suffix = self.EXCHANGE_MAP.get(exchange)
            if exchange_suffix:
                # 代码格式：600000, 000001
                if exchange_suffix == "sh":
                    # 上海: 6xxxxx
                    df_info = df_info[df_info['code'].str.startswith('6')]
                else:  # sz
                    # 深圳: 0xxxxx 或 3xxxxx
                    df_info = df_info[df_info['code'].str.match(r'^[03]')]

            # 转换为标准格式
            df = pd.DataFrame({
                'ts_code': df_info['code'].apply(lambda x: f"{x}.{self._exchange_to_value(exchange)}"),
                'symbol': df_info['code'],
                'name': df_info['name'],
                'area': '',
                'industry': '',
                'market': self._get_market_type(exchange),
                'list_date': ''
            })

            if df.empty:
                logger.warning(f"No stocks found for exchange: {exchange.value}")
            else:
                logger.info(f"Retrieved {len(df)} stocks from {exchange.value}")

            # 添加延迟
            time.sleep(self.delay)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch stock list from {exchange.value}: {e}")
            raise DataFetchError(f"Stock list fetch failed: {e}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.ConnectionError))
    )
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
        try:
            # 解析股票代码
            symbol, exchange = self._parse_ts_code(ts_code)
            exchange_suffix = self.EXCHANGE_MAP.get(Exchange(exchange))

            if not exchange_suffix:
                raise DataFetchError(f"Unsupported exchange: {exchange}")

            logger.debug(f"Fetching daily price for {ts_code} ({start_date} ~ {end_date})")

            # 转换日期格式 (YYYYMMDD -> YYYY-MM-DD)
            start_date_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_date_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

            # AkShare 获取历史数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date_str,
                end_date=end_date_str,
                adjust="qfq"  # 前复权
            )

            if df.empty:
                logger.warning(
                    f"No data for {ts_code} from {start_date} to {end_date}. "
                    f"Possible reasons: delisted, suspended, or invalid date range."
                )
                return pd.DataFrame()

            # 转换为标准格式
            result = pd.DataFrame({
                'ts_code': ts_code,
                'trade_date': df['日期'].dt.strftime('%Y%m%d'),
                'open': df['开盘'],
                'high': df['最高'],
                'low': df['最低'],
                'close': df['收盘'],
                'volume': df['成交量'],
                'amount': df['成交额'] / 1000  # 转换为千元
            })

            logger.debug(f"Retrieved {len(result)} records for {ts_code}")

            # 添加延迟
            time.sleep(self.delay)

            return result

        except Exception as e:
            logger.error(f"Failed to fetch daily price for {ts_code}: {e}")
            raise DataFetchError(f"Daily price fetch failed: {e}")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.ConnectionError))
    )
    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线数据

        常用指数代码：
        - 000001.SH: 上证综指 (sh000001)
        - 399001.SZ: 深证成指 (sz399001)
        - 399006.SZ: 创业板指 (sz399006)
        - 000300.SH: 沪深300 (sh000300)
        - 000016.SH: 上证50 (sh000016)
        - 000905.SH: 中证500 (sh000905)

        Args:
            index_code: 指数代码（如 000001.SH）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            指数数据 DataFrame，列结构同日线数据

        Raises:
            DataFetchError: 数据获取失败
        """
        try:
            # 解析指数代码
            symbol, exchange = self._parse_ts_code(index_code)

            logger.debug(f"Fetching index data for {index_code} ({start_date} ~ {end_date})")

            # 转换日期格式
            start_date_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_date_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

            # AkShare 获取指数历史数据
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date_str,
                end_date=end_date_str
            )

            if df.empty:
                logger.warning(f"No data for index {index_code}")
                return pd.DataFrame()

            # 转换为标准格式
            result = pd.DataFrame({
                'ts_code': index_code,
                'trade_date': df['日期'].dt.strftime('%Y%m%d'),
                'open': df['开盘'],
                'high': df['最高'],
                'low': df['最低'],
                'close': df['收盘'],
                'volume': df['成交量'],
                'amount': df['成交额'] / 1000
            })

            logger.debug(f"Retrieved {len(result)} records for index {index_code}")

            # 添加延迟
            time.sleep(self.delay)

            return result

        except Exception as e:
            logger.error(f"Failed to fetch index data for {index_code}: {e}")
            raise DataFetchError(f"Index data fetch failed: {e}")

    def _format_ts_code(self, code: str) -> str:
        """
        格式化股票代码为标准格式

        Args:
            code: 原始代码（如 sh600000）

        Returns:
            标准格式代码（如 600000.SH）
        """
        code_lower = code.lower()
        if code_lower.startswith('sh') or code_lower.startswith('bj'):
            exchange = code_lower[:2].upper()
            symbol = code[2:]
            return f"{symbol}.{Exchange(exchange).value}"
        elif code_lower.startswith('sz'):
            return f"{code[2:]}.SZSE"
        return code

    def _parse_ts_code(self, ts_code: str) -> tuple:
        """
        解析标准格式股票代码

        Args:
            ts_code: 标准格式代码（如 600000.SH）

        Returns:
            (symbol, exchange) 如 ("600000", "SSE")
        """
        parts = ts_code.split('.')
        if len(parts) == 2:
            return parts[0], parts[1]
        return ts_code, ""

    def _exchange_to_value(self, exchange: Exchange) -> str:
        """
        将交易所枚举转换为值

        Args:
            exchange: 交易所枚举

        Returns:
            交易所值字符串
        """
        return exchange.value

    def _get_market_type(self, exchange: Exchange) -> str:
        """
        获取市场类型

        Args:
            exchange: 交易所枚举

        Returns:
            市场类型字符串
        """
        market_map = {
            Exchange.SSE: "主板",
            Exchange.SZSE: "主板",
            Exchange.BSE: "北交所",
        }
        return market_map.get(exchange, "主板")
