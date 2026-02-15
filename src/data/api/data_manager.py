"""
数据管理器 - 门面类

提供统一的数据访问接口，整合获取、缓存、存储

注意：系统中存在双数据源问题
- DataManager（本模块）: 使用 CSV DataStorage（src.data.storage.storage）
- API服务（src.api.services.*）: 使用 SQLiteStorage（src.data.storage.sqlite_storage）

这可能导致：
1. 数据不一致：回测使用CSV数据，API使用SQLite数据
2. 数据同步困难：两个存储之间没有同步机制

TODO: 统一数据源，建议全部迁移到SQLiteStorage，DataManager也使用SQLite
"""

import os
from typing import Union, List
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

from src.data.fetchers.base import BaseDataFetcher, Exchange
from src.data.storage.storage import DataStorage
from src.data.cache.cache import DataCache


class DataManager:
    """
    数据管理器 - 门面类（Facade Pattern）

    职责：
    1. 协调各个组件（fetcher, storage, cache）
    2. 提供统一的高级 API
    3. 自动处理数据获取、缓存、存储流程

    数据访问优先级：
    1. 内存缓存 → 2. 本地文件 → 3. API 获取

    使用示例：
        # 使用 Mock 数据
        fetcher = MockDataFetcher()
        storage = DataStorage()
        manager = DataManager(fetcher=fetcher, storage=storage)

        # 获取数据
        df = manager.get_daily_price("600000.SH", "20230101", "20231231")

        # 批量下载
        manager.fetch_and_store(Exchange.SSE, "20230101", "20231231")
    """

    def __init__(
        self,
        fetcher: BaseDataFetcher,
        storage: DataStorage,
        cache_size: int = 100
    ):
        """
        初始化数据管理器

        Args:
            fetcher: 数据获取器（Mock/Tushare/AkShare）
            storage: 数据存储管理器
            cache_size: 缓存大小（最近访问的股票数）
        """
        self.fetcher = fetcher
        self.storage = storage
        self.cache = DataCache(size=cache_size)

        logger.info(
            f"DataManager initialized: "
            f"fetcher={type(fetcher).__name__}, "
            f"cache_size={cache_size}"
        )

    def get_daily_price(
        self,
        ts_code: str,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取日线数据（优先从缓存读取）

        优先级：
        1. 内存缓存
        2. 本地文件
        3. API 获取（需要 start_date 和 end_date）

        Args:
            ts_code: 股票代码
            start_date: 开始日期（YYYYMMDD），可选
            end_date: 结束日期（YYYYMMDD），可选
            use_cache: 是否使用缓存（默认 True）

        Returns:
            日线数据 DataFrame

        Raises:
            ValueError: 数据不存在且未提供日期范围

        Examples:
            >>> # 获取全部数据（从本地或缓存）
            >>> df = manager.get_daily_price("600000.SH")

            >>> # 获取指定日期范围
            >>> df = manager.get_daily_price("600000.SH", "20230101", "20231231")
        """
        # 1. 检查缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}"
        if use_cache and self.cache.exists(cache_key):
            logger.debug(f"Cache hit for {ts_code}")
            df = self.cache.get(cache_key)

            # 过滤日期范围
            if start_date or end_date:
                df = self._filter_by_date(df, start_date, end_date)

            return df

        # 2. 从本地加载
        exchange = self._get_exchange_from_code(ts_code)
        if self.storage.exists(ts_code, exchange):
            logger.debug(f"Loading from local: {ts_code}")
            df = self.storage.load_daily_price(ts_code, exchange)

            # 更新缓存
            self.cache.put(cache_key, df)

            # 过滤日期范围
            if start_date or end_date:
                df = self._filter_by_date(df, start_date, end_date)

            return df

        # 3. 从 API 获取
        if not start_date or not end_date:
            raise ValueError(
                f"Data not found for {ts_code}. "
                f"Please provide start_date and end_date for API fetching."
            )

        logger.info(f"Fetching from API: {ts_code} ({start_date} ~ {end_date})")

        try:
            df = self.fetcher.get_daily_price(ts_code, start_date, end_date)

            if df.empty:
                logger.warning(f"API returned empty data for {ts_code}")
                return pd.DataFrame()

            # 保存到本地
            self.storage.save_daily_price(df, ts_code, exchange)

            # 更新缓存
            self.cache.put(cache_key, df)

            logger.info(f"Successfully fetched and saved {len(df)} records for {ts_code}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {ts_code}: {e}")
            raise

    def get_stock_list(self, exchange: Exchange) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            exchange: 交易所枚举

        Returns:
            股票列表 DataFrame

        Examples:
            >>> df = manager.get_stock_list(Exchange.SSE)
            >>> print(df.head())
               ts_code   name  industry
            0  600000.SH  浦发银行      金融
        """
        try:
            df = self.fetcher.get_stock_list(exchange)

            # 保存到元数据目录
            if not df.empty:
                self.storage.save_stock_list(df, exchange)

            return df

        except Exception as e:
            logger.error(f"Failed to get stock list for {exchange.value}: {e}")
            raise

    def fetch_and_store(
        self,
        exchange: Exchange,
        start_date: str,
        end_date: str,
        force_update: bool = False
    ) -> None:
        """
        批量获取并存储数据

        适用于全市场数据下载

        Args:
            exchange: 交易所
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            force_update: 是否强制更新（覆盖已有数据）

        Examples:
            >>> # 下载上交所 2023 年数据
            >>> manager.fetch_and_store(Exchange.SSE, "20230101", "20231231")
        """
        # 获取股票列表
        stock_list = self.get_stock_list(exchange)

        if stock_list.empty:
            logger.warning(f"No stocks found for {exchange.value}")
            return

        total = len(stock_list)

        logger.info(
            f"Starting batch download: {total} stocks from {exchange.value} "
            f"({start_date} ~ {end_date})"
        )

        success_count = 0
        failed_count = 0
        skipped_count = 0

        # 遍历每只股票
        for idx, row in stock_list.iterrows():
            ts_code = row["ts_code"]

            try:
                # 跳过已存在的数据（除非强制更新）
                if not force_update and self.storage.exists(ts_code, exchange):
                    logger.debug(f"Skipping {ts_code} (already exists)")
                    skipped_count += 1
                    continue

                # 获取数据
                df = self.fetcher.get_daily_price(ts_code, start_date, end_date)

                if df.empty:
                    logger.warning(f"No data for {ts_code}")
                    failed_count += 1
                    continue

                # 保存数据
                self.storage.save_daily_price(df, ts_code, exchange)

                success_count += 1

                # 每 10 只股票输出一次进度
                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Progress: [{idx+1}/{total}] "
                        f"Success={success_count}, Skipped={skipped_count}, Failed={failed_count}"
                    )

            except Exception as e:
                logger.error(f"Failed to fetch {ts_code}: {e}")
                failed_count += 1

        logger.info(
            f"Batch download completed: "
            f"Total={total}, Success={success_count}, "
            f"Skipped={skipped_count}, Failed={failed_count}"
        )

    def update_latest(self, exchange: Exchange, days: int = 30) -> None:
        """
        增量更新最新数据

        只获取最近的数据，用于日常更新

        Args:
            exchange: 交易所
            days: 获取最近多少天的数据（默认 30）

        Examples:
            >>> # 更新最近 30 天数据
            >>> manager.update_latest(Exchange.SSE)
        """
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        logger.info(f"Updating latest data for {exchange.value} (last {days} days)")

        self.fetch_and_store(exchange, start_date, end_date, force_update=False)

    def get_cache_stats(self) -> dict:
        """
        获取缓存统计

        Returns:
            缓存统计信息字典
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("Cache cleared")

    # ========== 私有方法 ==========

    @staticmethod
    def _get_exchange_from_code(ts_code: str) -> Exchange:
        """
        从股票代码提取交易所

        Examples:
            600000.SH -> Exchange.SSE
            000001.SZ -> Exchange.SZSE
            836079.BJ -> Exchange.BSE

        Args:
            ts_code: 股票代码

        Returns:
            交易所枚举

        Raises:
            ValueError: 未知的交易所后缀
        """
        suffix = ts_code.split(".")[-1] if "." in ts_code else ""

        if suffix == "SH":
            return Exchange.SSE
        elif suffix == "SZ":
            return Exchange.SZSE
        elif suffix == "BJ":
            return Exchange.BSE
        else:
            raise ValueError(f"Unknown exchange suffix in ts_code: {ts_code}")

    @staticmethod
    def _filter_by_date(
        df: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        按日期范围过滤

        Args:
            df: 数据 DataFrame
            start_date: 开始日期（YYYYMMDD 或 datetime）
            end_date: 结束日期（YYYYMMDD 或 datetime）

        Returns:
            过滤后的 DataFrame
        """
        if df.empty or "trade_date" not in df.columns:
            return df

        # 确保日期列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

        # 转换过滤条件为 datetime
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date, format="%Y%m%d")
            df = df[df["trade_date"] >= start_date]

        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date, format="%Y%m%d")
            df = df[df["trade_date"] <= end_date]

        return df
