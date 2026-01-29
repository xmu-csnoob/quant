"""
增强版机器学习特征提取器

包含60+个特征，涵盖：
1. 原有技术指标（29个）
2. 时间特征（8个）
3. 价格模式特征（10个）
4. 成交量模式特征（8个）
5. 统计特征（5个）
"""

import pandas as pd
import numpy as np
from typing import List
from loguru import logger

from utils.features.ml_features import MLFeatureExtractor


class EnhancedFeatureExtractor:
    """
    增强版特征提取器

    在原有技术指标基础上，添加：
    - 时间日历特征
    - 价格形态特征
    - 成交量形态特征
    - 统计分布特征
    """

    def __init__(self, prediction_period: int = 5):
        self.prediction_period = prediction_period
        self.base_extractor = MLFeatureExtractor(prediction_period)

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有增强特征

        Args:
            df: 原始OHLCV数据

        Returns:
            添加了所有特征的DataFrame
        """
        df = df.copy()

        # 确保日期是datetime
        if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

        # 1. 基础技术指标（29个）
        df = self.base_extractor.extract(df)

        # 2. 时间特征
        df = self._add_time_features(df)

        # 3. 价格模式特征
        df = self._add_price_pattern_features(df)

        # 4. 成交量模式特征
        df = self._add_volume_pattern_features(df)

        # 5. 统计特征
        df = self._add_statistical_features(df)

        feature_count = len([c for c in df.columns if c.startswith("f_")])
        logger.info(f"提取了 {feature_count} 个增强特征")

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间日历特征"""
        # 星期几（0=周一, 4=周五）
        df["f_day_of_week"] = df["trade_date"].dt.dayofweek / 4.0  # 归一化

        # 月份
        df["f_month"] = (df["trade_date"].dt.month - 1) / 11.0

        # 季度
        df["f_quarter"] = (df["trade_date"].dt.quarter - 1) / 3.0

        # 月初（1-5日）
        df["f_month_start"] = (df["trade_date"].dt.day <= 5).astype(int)

        # 月末（25日以后）
        df["f_month_end"] = (df["trade_date"].dt.day >= 25).astype(int)

        # 年内天数
        df["f_day_of_year"] = df["trade_date"].dt.dayofyear / 365.0

        return df

    def _add_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格形态特征"""
        # 跳空（开盘价相对前日收盘价的跳空幅度）
        df["f_gap"] = (df["open"] / df["close"].shift(1) - 1).fillna(0)

        # 振幅（(最高-最低)/开盘）
        df["f_daily_range"] = (df["high"] - df["low"]) / df["open"]

        # 影线比例
        body = abs(df["close"] - df["open"])
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        df["f_upper_shadow_ratio"] = upper_shadow / (df["high"] - df["low"])
        df["f_lower_shadow_ratio"] = lower_shadow / (df["high"] - df["low"])

        # 连续上涨/下跌天数
        price_change = np.sign(df["close"].diff())
        df["f_consecutive_up"] = (price_change > 0).astype(int).rolling(5).sum()
        df["f_consecutive_down"] = (price_change < 0).astype(int).rolling(5).sum()

        # 近期涨跌幅标准差（波动率）
        df["f_return_std_10"] = df["close"].pct_change().rolling(10).std()

        # 价格位置（近20天中的分位数）- 使用前一日价格避免前视偏差
        df["f_price_percentile_20"] = df["close"].shift(1).rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )

        # 前日涨跌
        df["f_yesterday_return"] = df["close"].pct_change(1)

        return df

    def _add_volume_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量形态特征"""
        # 量比（当日成交量 / 5日/20日/60日均线）
        for period in [5, 20, 60]:
            vol_ma = df["volume"].rolling(period).mean()
            df[f"f_volume_ratio_{period}"] = df["volume"] / vol_ma

        # 放量（成交量 > 2倍5日均量）
        vol_ma5 = df["volume"].rolling(5).mean()
        df["f_volume_surge"] = (df["volume"] > 2 * vol_ma5).astype(int)

        # 缩量（成交量 < 0.5倍5日均量）
        df["f_volume_shrink"] = (df["volume"] < 0.5 * vol_ma5).astype(int)

        # 成交量变化率
        df["f_volume_change"] = df["volume"].pct_change(5)

        # 量价背离（价格上涨但成交量下降）
        price_up = df["close"] > df["close"].shift(5)
        vol_down = df["volume"] < df["volume"].shift(5)
        df["f_divergence"] = (price_up & vol_down).astype(int)

        # 成交额占比（成交额 / 开盘价×成交量）
        df["f_amount_ratio"] = df["amount"] / (df["open"] * df["volume"])

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计分布特征"""
        returns = df["close"].pct_change()

        # 偏度（skewness）- 衡量收益分布的对称性
        df["f_return_skew_20"] = returns.rolling(20).skew()

        # 峰度（kurtosis）- 衡量收益分布的尾部厚度
        df["f_return_kurt_20"] = returns.rolling(20).kurt()

        # 正收益比例（近20天）
        df["f_positive_ratio_20"] = (returns > 0).rolling(20).mean()

        # 累计收益（近5/10/20天）
        for period in [5, 10, 20]:
            df[f"f_cum_return_{period}"] = returns.rolling(period).apply(
                lambda x: (1 + x).prod() - 1
            )

        return df

    def get_feature_count(self, df: pd.DataFrame) -> int:
        """获取特征数量"""
        return len([c for c in df.columns if c.startswith("f_")])
