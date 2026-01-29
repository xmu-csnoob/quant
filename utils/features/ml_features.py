"""
机器学习特征提取器

为ML模型准备40+个技术特征
"""

import pandas as pd
import numpy as np
from typing import List
from loguru import logger


class MLFeatureExtractor:
    """
    机器学习特征提取器

    特征分类：
    1. 价格动量特征 (4个)
    2. 技术指标特征 (8个)
    3. 均线系统特征 (4个)
    4. 成交量特征 (5个)
    5. 波动率特征 (3个)
    6. 交互特征 (~10个)
    """

    def __init__(self, prediction_period: int = 5):
        """
        初始化

        Args:
            prediction_period: 预测周期（天数）
        """
        self.prediction_period = prediction_period

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有ML特征

        Args:
            df: 原始OHLCV数据，需包含列：
                - open, high, low, close, volume
                - trade_date

        Returns:
            添加了特征列的DataFrame
        """
        df = df.copy()

        # 确保日期是datetime
        if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

        # 1. 价格动量特征
        df = self._add_momentum_features(df)

        # 2. 技术指标特征
        df = self._add_indicator_features(df)

        # 3. 均线系统特征
        df = self._add_ma_features(df)

        # 4. 成交量特征
        df = self._add_volume_features(df)

        # 5. 波动率特征
        df = self._add_volatility_features(df)

        # 6. 交互特征
        df = self._add_interaction_features(df)

        logger.info(f"提取了 {len([c for c in df.columns if c.startswith('f_')])} 个ML特征")

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格动量特征"""
        # 各种周期的收益率
        for period in [1, 3, 5, 10]:
            df[f"f_return_{period}d"] = df["close"].pct_change(period)

        # 动量强弱
        df["f_momentum_5_20"] = (
            df["close"].pct_change(5) - df["close"].pct_change(20)
        )

        return df

    def _add_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """技术指标特征"""
        # RSI (14日)
        df["f_rsi"] = self._calculate_rsi(df["close"], 14)

        # MACD
        macd_line = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        df["f_macd"] = macd_line - signal_line
        df["f_macd_hist"] = macd_line - signal_line

        # 布林带位置（价格在布林带中的位置）
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df["f_bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

        # ATR (14日)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["f_atr"] = tr.rolling(14).mean()
        df["f_atr_ratio"] = df["f_atr"] / df["close"]

        return df

    def _add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """均线系统特征"""
        # 价格相对均线的位置 - 使用昨日价格避免前视偏差
        close_yesterday = df["close"].shift(1)
        for period in [5, 10, 20, 60]:
            ma = close_yesterday.rolling(period).mean()
            df[f"f_ma_{period}_ratio"] = close_yesterday / ma - 1

        # 均线斜率
        ma20 = df["close"].rolling(20).mean()
        df["f_ma20_slope"] = ma20.pct_change(5)

        # 多头排列（短期>长期）
        ma5 = df["close"].rolling(5).mean()
        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()
        df["f_ma_alignment"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特征"""
        # 量比（当日成交量 / 5日均量）
        vol_ma5 = df["volume"].rolling(5).mean()
        df["f_volume_ratio"] = df["volume"] / vol_ma5

        # 量价背离
        price_change = df["close"].pct_change()
        volume_change = df["volume"].pct_change()
        df["f_price_volume_trend"] = price_change * volume_change

        # OBV (能量潮)
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["f_obv"] = obv
        df["f_obv_ma"] = obv.rolling(20).mean()
        df["f_obv_ratio"] = obv / df["f_obv_ma"]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率特征"""
        # 历史波动率（标准差）
        returns = df["close"].pct_change()
        df["f_volatility_10"] = returns.rolling(10).std()
        df["f_volatility_20"] = returns.rolling(20).std()

        # 波动率变化
        df["f_volatility_change"] = df["f_volatility_10"] / df["f_volatility_20"]

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互特征"""
        # RSI × 趋势
        ma20_ratio = df["close"] / df["close"].rolling(20).mean() - 1
        df["f_rsi_trend"] = df["f_rsi"] * np.sign(ma20_ratio)

        # 波动率 × 动量
        df["f_vol_momentum"] = df["f_volatility_10"] * df["f_return_5d"]

        # 成交量 × 价格变化
        df["f_vol_price"] = df["f_volume_ratio"] * df["f_return_5d"]

        # ATR × RSI
        df["f_atr_rsi"] = df["f_atr_ratio"] * (df["f_rsi"] / 100)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_feature_names(self) -> List[str]:
        """
        获取所有特征名

        Returns:
            特征名列表（以f_开头的列）
        """
        # 这个方法需要在extract()后调用才有意义
        # 这里返回一个示例列表
        features = []
        for p in [1, 3, 5, 10]:
            features.append(f"f_return_{p}d")
        features.extend([
            "f_momentum_5_20",
            "f_rsi", "f_macd", "f_macd_hist", "f_bb_position",
            "f_atr", "f_atr_ratio",
            "f_ma_5_ratio", "f_ma_10_ratio", "f_ma_20_ratio", "f_ma_60_ratio",
            "f_ma20_slope", "f_ma_alignment",
            "f_volume_ratio", "f_price_volume_trend", "f_obv", "f_obv_ma", "f_obv_ratio",
            "f_volatility_10", "f_volatility_20", "f_volatility_change",
            "f_rsi_trend", "f_vol_momentum", "f_vol_price", "f_atr_rsi",
        ])
        return features
