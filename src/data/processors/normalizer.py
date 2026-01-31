"""
数据标准化处理器

负责：
1. 字段名映射（不同数据源 → 统一字段名）
2. 数据类型统一（日期格式、数值类型）
3. 单位统一（成交额、成交量的单位）
4. 补充缺失字段（填充 NaN）
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional
from loguru import logger

from src.data.fetchers.base import DataValidationError


# ========================================
# 字段映射规则
# ========================================

# Tushare 字段映射
TUSHARE_FIELD_MAP = {
    "vol": "volume",  # Tushare 用 vol，我们用 volume
}

# AkShare 字段映射（已在其内部处理，这里保留扩展性）
AKSHARE_FIELD_MAP = {}

# 所有数据源统一到目标字段
TARGET_FIELDS = [
    "ts_code",           # 股票代码
    "trade_date",        # 交易日期 YYYYMMDD
    "open",              # 开盘价
    "high",              # 最高价
    "low",               # 最低价
    "close",             # 收盘价
    "pre_close",         # 昨收价
    "change",            # 涨跌额
    "pct_chg",           # 涨跌幅 (%)
    "volume",            # 成交量（手）
    "amount",            # 成交额（千元）
    "turnover_rate",     # 换手率 (%)
    "pe_ttm",            # 市盈率 TTM
    "pb_mrq",            # 市净率
    "total_mv",          # 总市值（万元）
    "circ_mv",           # 流通市值（万元）
]

# 股票列表字段
STOCK_LIST_FIELDS = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "list_date",
    "exchange",  # 新增：交易所
]


# ========================================
# 标准化器
# ========================================

class DataNormalizer:
    """数据标准化器

    将不同数据源的数据统一到标准格式
    """

    def __init__(
        self,
        source: Literal["tushare", "akshare", "mock"],
        fill_missing: bool = True
    ):
        """初始化标准化器

        Args:
            source: 数据源类型
            fill_missing: 是否填充缺失字段为 NaN
        """
        self.source = source.lower()
        self.fill_missing = fill_missing

        # 选择字段映射规则
        if self.source == "tushare":
            self.field_map = TUSHARE_FIELD_MAP
        elif self.source == "akshare":
            self.field_map = AKSHARE_FIELD_MAP
        elif self.source == "mock":
            self.field_map = {}
        else:
            raise ValueError(f"Unknown data source: {source}")

        logger.debug(f"DataNormalizer initialized for source: {source}")

    def normalize_daily_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化日线数据

        Args:
            df: 原始数据

        Returns:
            标准化后的 DataFrame

        Raises:
            DataValidationError: 数据验证失败
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to normalize")
            return df

        # 1. 字段名映射
        df = df.rename(columns=self.field_map)

        # 2. 验证必需字段
        required_fields = ["ts_code", "trade_date", "close"]
        missing = [f for f in required_fields if f not in df.columns]
        if missing:
            raise DataValidationError(
                f"Missing required fields: {missing}. "
                f"Available fields: {df.columns.tolist()}"
            )

        # 3. 日期格式统一 (YYYYMMDD)
        if "trade_date" in df.columns:
            df["trade_date"] = self._normalize_date(df["trade_date"])

        # 4. 数值类型转换
        numeric_fields = [
            "open", "high", "low", "close", "pre_close",
            "change", "pct_chg", "volume", "amount",
            "turnover_rate", "pe_ttm", "pb_mrq", "total_mv", "circ_mv"
        ]
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors="coerce")

        # 5. 单位统一
        df = self._normalize_units(df)

        # 6. 补充缺失字段
        if self.fill_missing:
            df = self._fill_missing_fields(df, TARGET_FIELDS)

        # 7. 交易所信息
        df = self._add_exchange_info(df)

        logger.debug(
            f"Normalized {len(df)} rows from {self.source}. "
            f"Columns: {df.columns.tolist()}"
        )

        return df

    def normalize_stock_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化股票列表数据

        Args:
            df: 原始数据

        Returns:
            标准化后的 DataFrame
        """
        if df.empty:
            return df

        # 1. 字段名映射
        df = df.rename(columns=self.field_map)

        # 2. 日期格式统一
        if "list_date" in df.columns:
            df["list_date"] = self._normalize_date(df["list_date"])

        # 3. 补充缺失字段
        if self.fill_missing:
            df = self._fill_missing_fields(df, STOCK_LIST_FIELDS)

        # 4. 交易所信息
        df = self._add_exchange_info(df)

        logger.debug(f"Normalized stock list: {len(df)} stocks")

        return df

    def normalize_index_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化指数日线数据

        Args:
            df: 原始数据

        Returns:
            标准化后的 DataFrame
        """
        if df.empty:
            return df

        # 指数数据与股票数据结构类似，重命名后使用相同逻辑
        df = df.rename(columns={"index_code": "ts_code"})

        # 使用日线数据标准化逻辑
        result = self.normalize_daily_price(df)

        # 恢复 index_code
        if "ts_code" in result.columns:
            result = result.rename(columns={"ts_code": "index_code"})

        return result

    # ========================================
    # 内部辅助方法
    # ========================================

    def _normalize_date(self, series: pd.Series) -> pd.Series:
        """统一日期格式为 YYYYMMDD

        Args:
            series: 日期序列

        Returns:
            格式化后的日期序列
        """
        # 如果已经是 YYYYMMDD 格式的字符串，直接返回
        if series.dtype == "object":
            # 检查是否已经是 8 位数字
            sample = series.iloc[0] if len(series) > 0 else ""
            if isinstance(sample, str) and len(sample) == 8 and sample.isdigit():
                return series

        # 尝试解析并转换
        try:
            # 如果是 datetime 类型
            if pd.api.types.is_datetime64_any_dtype(series):
                return series.dt.strftime("%Y%m%d")

            # 如果是字符串但格式不同
            return pd.to_datetime(series).dt.strftime("%Y%m%d")

        except Exception as e:
            logger.warning(f"Date normalization warning: {e}")
            return series

    def _normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一数据单位

        标准单位：
        - volume: 手（1 手 = 100 股）
        - amount: 千元

        Args:
            df: 数据

        Returns:
            单位统一后的数据
        """
        # Tushare 的 amount 单位已经是千元，无需转换
        # AkShare 在获取时已转换
        # 这里主要做验证和日志记录

        return df

    def _fill_missing_fields(
        self,
        df: pd.DataFrame,
        target_fields: list[str]
    ) -> pd.DataFrame:
        """补充缺失的字段

        Args:
            df: 数据
            target_fields: 目标字段列表

        Returns:
            补充字段后的数据
        """
        for field in target_fields:
            if field not in df.columns:
                df[field] = np.nan
                logger.debug(f"Added missing field: {field}")

        # 重新排列列顺序
        existing_fields = [f for f in target_fields if f in df.columns]
        other_fields = [f for f in df.columns if f not in target_fields]

        return df[existing_fields + other_fields]

    def _add_exchange_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交易所信息

        从 ts_code 解析交易所

        Args:
            df: 数据

        Returns:
            添加交易所后的数据
        """
        if "ts_code" not in df.columns:
            return df

        # 如果已经有有效的 exchange 字段，跳过
        if "exchange" in df.columns and not df["exchange"].isna().all():
            return df

        def parse_exchange(ts_code: str) -> str:
            """从股票代码解析交易所"""
            if isinstance(ts_code, str):
                suffix = ts_code.split(".")[-1] if "." in ts_code else ""
                if suffix == "SH":
                    return "SSE"
                elif suffix == "SZ":
                    return "SZSE"
                elif suffix == "BJ":
                    return "BSE"
            return ""

        df["exchange"] = df["ts_code"].apply(parse_exchange)

        return df


# ========================================
# 便捷函数
# ========================================

def normalize_from_source(
    df: pd.DataFrame,
    source: Literal["tushare", "akshare", "mock"],
    data_type: Literal["daily", "stock_list", "index_daily"] = "daily"
) -> pd.DataFrame:
    """便捷函数：标准化数据

    Args:
        df: 原始数据
        source: 数据源类型
        data_type: 数据类型

    Returns:
        标准化后的数据
    """
    normalizer = DataNormalizer(source=source)

    if data_type == "daily":
        return normalizer.normalize_daily_price(df)
    elif data_type == "stock_list":
        return normalizer.normalize_stock_list(df)
    elif data_type == "index_daily":
        return normalizer.normalize_index_daily(df)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
