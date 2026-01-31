"""
数据存储模块

管理数据的本地存储和加载
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from src.data.fetchers.base import Exchange


class DataStorage:
    """
    数据存储管理

    特点：
    1. 按交易所和股票代码组织存储
    2. 自动创建目录结构
    3. CSV 格式存储（兼容性好）
    4. 支持增量更新

    目录结构：
    data/
    ├── raw/               # 原始数据
    │   ├── sse/
    │   │   ├── stocks/daily/
    │   │   └── index/
    │   ├── szse/
    │   └── bse/
    ├── processed/         # 处理后数据
    │   ├── adjusted/
    │   ├── indicators/
    │   └── factors/
    └── metadata/          # 元数据
        ├── stock_list.json
        └── trading_calendar.json
    """

    def __init__(
        self,
        base_path: str = "data/raw",
        processed_path: str = "data/processed",
        metadata_path: str = "data/metadata",
        encoding: str = "utf-8"
    ):
        """
        初始化数据存储

        Args:
            base_path: 原始数据存储路径
            processed_path: 处理后数据存储路径
            metadata_path: 元数据存储路径
            encoding: CSV 文件编码
        """
        self.base_path = Path(base_path)
        self.processed_path = Path(processed_path)
        self.metadata_path = Path(metadata_path)
        self.encoding = encoding

        # 创建目录结构
        self._create_directory_structure()

        logger.info(
            f"DataStorage initialized: "
            f"raw={base_path}, processed={processed_path}, metadata={metadata_path}"
        )

    def _create_directory_structure(self) -> None:
        """创建必要的目录结构"""
        directories = [
            # 原始数据目录
            self.base_path / "sse" / "stocks" / "daily",
            self.base_path / "sse" / "index",
            self.base_path / "szse" / "stocks" / "daily",
            self.base_path / "szse" / "index",
            self.base_path / "bse" / "stocks" / "daily",
            self.base_path / "bse" / "index",
            # 处理后数据目录
            self.processed_path / "adjusted",
            self.processed_path / "indicators",
            self.processed_path / "factors",
            # 元数据目录
            self.metadata_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.debug("Directory structure created")

    def _get_file_path(
        self,
        ts_code: str,
        exchange: Exchange,
        data_type: str = "daily"
    ) -> Path:
        """
        获取数据文件路径

        Args:
            ts_code: 股票代码
            exchange: 交易所
            data_type: 数据类型（daily/index）

        Returns:
            文件路径对象

        Examples:
            >>> _get_file_path("600000.SH", Exchange.SSE)
            Path("data/raw/sse/stocks/daily/600000.SH.csv")
        """
        exchange_dir = exchange.value.lower()

        if data_type == "daily":
            return (
                self.base_path
                / exchange_dir
                / "stocks"
                / "daily"
                / f"{ts_code}.csv"
            )
        elif data_type == "index":
            return (
                self.base_path
                / exchange_dir
                / "index"
                / f"{ts_code}.csv"
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def save_daily_price(
        self,
        df: pd.DataFrame,
        ts_code: str,
        exchange: Exchange
    ) -> None:
        """
        保存日线数据到 CSV

        Args:
            df: 日线数据 DataFrame
            ts_code: 股票代码
            exchange: 交易所

        Raises:
            IOError: 文件写入失败
        """
        try:
            file_path = self._get_file_path(ts_code, exchange, "daily")

            # 按交易日期排序
            df = df.sort_values("trade_date")

            # 保存到 CSV
            df.to_csv(file_path, index=False, encoding=self.encoding)

            logger.debug(f"Saved {len(df)} rows to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save daily data for {ts_code}: {e}")
            raise IOError(f"Save failed: {e}")

    def load_daily_price(
        self,
        ts_code: str,
        exchange: Exchange
    ) -> pd.DataFrame:
        """
        从 CSV 加载日线数据

        Args:
            ts_code: 股票代码
            exchange: 交易所

        Returns:
            日线数据 DataFrame，如果文件不存在返回空 DataFrame
        """
        try:
            file_path = self._get_file_path(ts_code, exchange, "daily")

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return pd.DataFrame()

            df = pd.read_csv(file_path, encoding=self.encoding)

            # 标准化列名（兼容 Tushare 格式）
            if "vol" in df.columns and "volume" not in df.columns:
                df = df.rename(columns={"vol": "volume"})

            # 确保日期格式正确
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

            logger.debug(f"Loaded {len(df)} rows from {file_path}")

            return df

        except Exception as e:
            logger.error(f"Failed to load daily data for {ts_code}: {e}")
            return pd.DataFrame()

    def exists(self, ts_code: str, exchange: Exchange) -> bool:
        """
        检查数据文件是否存在

        Args:
            ts_code: 股票代码
            exchange: 交易所

        Returns:
            True 如果文件存在，否则 False
        """
        file_path = self._get_file_path(ts_code, exchange, "daily")
        return file_path.exists()

    def save_index_data(
        self,
        df: pd.DataFrame,
        index_code: str,
        exchange: Exchange
    ) -> None:
        """
        保存指数数据

        Args:
            df: 指数数据 DataFrame
            index_code: 指数代码
            exchange: 交易所
        """
        try:
            file_path = self._get_file_path(index_code, exchange, "index")

            # 按交易日期排序
            df = df.sort_values("trade_date")

            # 保存到 CSV
            df.to_csv(file_path, index=False, encoding=self.encoding)

            logger.debug(f"Saved {len(df)} rows to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save index data for {index_code}: {e}")
            raise IOError(f"Save failed: {e}")

    def load_index_data(
        self,
        index_code: str,
        exchange: Exchange
    ) -> pd.DataFrame:
        """
        加载指数数据

        Args:
            index_code: 指数代码
            exchange: 交易所

        Returns:
            指数数据 DataFrame，如果文件不存在返回空 DataFrame
        """
        try:
            file_path = self._get_file_path(index_code, exchange, "index")

            if not file_path.exists():
                logger.warning(f"Index file not found: {file_path}")
                return pd.DataFrame()

            df = pd.read_csv(file_path, encoding=self.encoding)

            # 确保日期格式正确
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

            logger.debug(f"Loaded {len(df)} rows from {file_path}")

            return df

        except Exception as e:
            logger.error(f"Failed to load index data for {index_code}: {e}")
            return pd.DataFrame()

    def save_stock_list(
        self,
        df: pd.DataFrame,
        exchange: Exchange
    ) -> None:
        """
        保存股票列表到元数据目录

        Args:
            df: 股票列表 DataFrame
            exchange: 交易所
        """
        try:
            file_path = self.metadata_path / f"stock_list_{exchange.value.lower()}.json"

            # 保存为 JSON
            df.to_json(file_path, orient="records", force_ascii=False, indent=2)

            logger.info(f"Saved stock list for {exchange.value} to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save stock list: {e}")
            raise IOError(f"Save failed: {e}")

    def load_stock_list(self, exchange: Exchange) -> pd.DataFrame:
        """
        加载股票列表

        Args:
            exchange: 交易所

        Returns:
            股票列表 DataFrame，如果文件不存在返回空 DataFrame
        """
        try:
            file_path = self.metadata_path / f"stock_list_{exchange.value.lower()}.json"

            if not file_path.exists():
                logger.warning(f"Stock list file not found: {file_path}")
                return pd.DataFrame()

            df = pd.read_json(file_path, orient="records")

            logger.info(f"Loaded {len(df)} stocks from {file_path}")

            return df

        except Exception as e:
            logger.error(f"Failed to load stock list: {e}")
            return pd.DataFrame()
