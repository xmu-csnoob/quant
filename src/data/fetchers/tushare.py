"""
Tushare 数据获取器

从 Tushare Pro API 获取真实的 A 股市场数据

注意：免费账户有频率限制
- 股票列表：每小时最多 1 次
- 分钟数据：每天最多 2 次
- 涨跌停数据：每小时最多 1 次
"""

import os
import tushare as ts
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import requests
from loguru import logger

from src.data.fetchers.base import (
    BaseDataFetcher,
    Exchange,
    TokenNotFoundError,
    DataFetchError,
    RateLimitError
)
from src.data.cache.persistent_cache import PersistentCache


class TushareDataFetcher(BaseDataFetcher):
    """
    Tushare 数据获取器

    特点：
    1. 官方 API，数据质量高
    2. 稳定可靠
    3. 支持全市场数据
    4. 需要申请 Token（有免费额度）
    5. 持久化缓存支持（应对频率限制）

    用途：
    - 生产环境数据获取
    - 真实数据验证
    - 历史数据回测

    Token 申请：
    https://tushare.pro/register
    """

    # 交易所代码映射
    EXCHANGE_MAP = {
        Exchange.SSE: "SSE",
        Exchange.SZSE: "SZSE",
        Exchange.BSE: "BSE",
    }

    # 缓存 TTL 配置（针对免费账户频率限制）
    CACHE_TTL = {
        "stock_list": 3600,      # 1小时（每小时限制1次）
        "minute_data": 86400,    # 24小时（每天限制2次）
        "limit_list": 3600,      # 1小时（每小时限制1次）
    }

    def __init__(
        self,
        token: str = None,
        max_retries: int = 3,
        cache_dir: str = "data/cache/tushare",
        proxy_url: str = None
    ):
        """
        初始化 Tushare 数据获取器

        Args:
            token: Tushare API Token
                   如果为 None，则从环境变量 TUSHARE_TOKEN 读取
            max_retries: 最大重试次数
            cache_dir: 缓存目录
            proxy_url: 代理URL（可选，用于破解版API）

        Raises:
            TokenNotFoundError: Token 为空或未设置环境变量
        """
        # 获取 Token
        if token is None:
            token = os.getenv("TUSHARE_TOKEN")
            # 尝试从环境变量获取代理
            if proxy_url is None:
                proxy_url = os.getenv("TUSHARE_PROXY_URL")

        if not token:
            raise TokenNotFoundError(
                "Tushare Token is required. "
                "Please set TUSHARE_TOKEN environment variable or pass token parameter. "
                "Token can be obtained from: https://tushare.pro/register"
            )

        # 设置 Token
        try:
            ts.set_token(token)
            self.pro = ts.pro_api()
            self.token = token
            self.max_retries = max_retries

            # 如果提供了代理URL，使用代理（破解版API）
            if proxy_url:
                self.pro._DataApi__token = token
                self.pro._DataApi__http_url = proxy_url
                logger.info(f"Using Tushare proxy: {proxy_url}")

            # 初始化持久化缓存
            self.cache = PersistentCache(cache_dir=cache_dir)

            logger.info("TushareDataFetcher initialized successfully with cache")

        except Exception as e:
            raise DataFetchError(f"Failed to initialize Tushare API: {e}")

    def _detect_rate_limit(self, error_message: str) -> tuple:
        """
        检测错误是否为频率限制

        Args:
            error_message: 错误消息

        Returns:
            (是否为频率限制, 限制类型)
        """
        # 检测每小时限制
        if "每小时最多访问" in error_message or "每小時最多訪問" in error_message:
            return True, "hourly"

        # 检测每天限制
        if "每天最多访问" in error_message or "每天最多訪問" in error_message:
            return True, "daily"

        # 检测其他可能的限制关键词
        msg_lower = error_message.lower()
        if "rate limit" in msg_lower or "too many" in msg_lower:
            return True, "unknown"

        return False, None

    def get_stock_list(self, exchange: Exchange) -> pd.DataFrame:
        """
        获取股票列表

        使用持久化缓存避免频率限制（每小时最多 1 次）

        Args:
            exchange: 交易所枚举

        Returns:
            股票列表 DataFrame

        Raises:
            DataFetchError: 数据获取失败
            RateLimitError: 频率限制（但会先尝试返回缓存）
        """
        # 生成缓存键
        cache_key = f"stock_list_{exchange.value}"

        # 先尝试从缓存获取
        cached_df = self.cache.get(cache_key)
        if cached_df is not None:
            logger.info(f"Retrieved {len(cached_df)} stocks from cache ({exchange.value})")
            return cached_df

        try:
            exchange_code = self.EXCHANGE_MAP[exchange]

            logger.debug(f"Fetching stock list from {exchange.value} (Tushare)")

            df = self.pro.stock_basic(
                exchange=exchange_code,
                list_status="L",  # 只获取上市股票
                fields="ts_code,symbol,name,area,industry,market,list_date"
            )

            if df.empty:
                logger.warning(f"No stocks found for exchange: {exchange.value}")
            else:
                logger.info(f"Retrieved {len(df)} stocks from {exchange.value}")
                # 保存到缓存
                self.cache.put(cache_key, df, ttl=self.CACHE_TTL["stock_list"])

            return df

        except Exception as e:
            error_msg = str(e)

            # 检测是否为频率限制
            is_rate_limit, limit_type = self._detect_rate_limit(error_msg)

            if is_rate_limit:
                logger.warning(f"Rate limit hit for stock list: {limit_type}")
                # 尝试返回过期的缓存（如果有）
                # 先删除当前元数据，尝试直接读取文件
                cache_file = self.cache._get_cache_file(cache_key)
                if cache_file.exists():
                    try:
                        import json
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if "_data" in data and "_columns" in data:
                            df = pd.DataFrame(data["_data"], columns=data["_columns"])
                            logger.warning(f"Using stale cache due to rate limit ({len(df)} rows)")
                            # 重新设置缓存
                            self.cache.put(cache_key, df, ttl=self.CACHE_TTL["stock_list"])
                            return df
                    except Exception:
                        pass

                # 抛出频率限制异常
                raise RateLimitError(
                    f"Stock list API rate limit exceeded: {error_msg}",
                    limit_type=limit_type
                )

            logger.error(f"Failed to fetch stock list from {exchange.value}: {e}")
            raise DataFetchError(f"Stock list fetch failed: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def get_daily_price(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取日线行情数据

        带重试机制，网络错误时自动重试

        Args:
            ts_code: 股票代码（如 600000.SH）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            日线数据 DataFrame

        Raises:
            DataFetchError: 数据获取失败
        """
        try:
            logger.debug(f"Fetching daily price for {ts_code} ({start_date} ~ {end_date})")

            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                logger.warning(
                    f"No data for {ts_code} from {start_date} to {end_date}. "
                    f"Possible reasons: delisted, suspended, or invalid date range."
                )
            else:
                logger.debug(f"Retrieved {len(df)} records for {ts_code}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch daily price for {ts_code}: {e}")
            raise DataFetchError(f"Daily price fetch failed: {e}")

    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指数日线数据

        常用指数代码：
        - 000001.SH: 上证综指
        - 399001.SZ: 深证成指
        - 399006.SZ: 创业板指
        - 000300.SH: 沪深300
        - 000016.SH: 上证50
        - 000905.SH: 中证500

        Args:
            index_code: 指数代码
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            指数数据 DataFrame

        Raises:
            DataFetchError: 数据获取失败
        """
        try:
            logger.debug(f"Fetching index data for {index_code} ({start_date} ~ {end_date})")

            df = self.pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                logger.warning(f"No data for index {index_code}")
            else:
                logger.debug(f"Retrieved {len(df)} records for index {index_code}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch index data for {index_code}: {e}")
            raise DataFetchError(f"Index data fetch failed: {e}")

    def get_trading_calendar(
        self,
        exchange: Exchange = Exchange.SSE,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取交易日历

        Args:
            exchange: 交易所
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            交易日历 DataFrame，包含列：
            - cal_date: 日历日期
            - is_open: 是否交易（1=是，0=否）

        Raises:
            DataFetchError: 数据获取失败
        """
        try:
            exchange_code = self.EXCHANGE_MAP[exchange]

            logger.debug(f"Fetching trading calendar for {exchange.value}")

            df = self.pro.trade_cal(
                exchange=exchange_code,
                start_date=start_date,
                end_date=end_date
            )

            logger.info(f"Retrieved {len(df)} calendar days")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch trading calendar: {e}")
            raise DataFetchError(f"Trading calendar fetch failed: {e}")

    def get_stock_basic_info(self, ts_code: str = None) -> pd.DataFrame:
        """
        获取股票基本信息

        Args:
            ts_code: 股票代码，如果为 None 则获取所有股票

        Returns:
            股票基本信息 DataFrame

        Raises:
            DataFetchError: 数据获取失败
        """
        try:
            logger.debug(f"Fetching basic info for {ts_code if ts_code else 'all stocks'}")

            df = self.pro.stock_basic(
                ts_code=ts_code,
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date"
            )

            logger.info(f"Retrieved basic info for {len(df)} stocks")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch stock basic info: {e}")
            raise DataFetchError(f"Stock basic info fetch failed: {e}")
