"""
技术指标特征提取器

提取所有技术指标相关的特征：
- 趋势特征：MA系列、MA斜率、多头排列
- 动能特征：MACD系列、动量、ROC
- 振荡特征：RSI、KDJ
- 波动率特征：布林带、ATR
- 成交量特征：成交量相关指标
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from .base import BaseFeatureExtractor
from ..indicators import MA, MACD, RSI


class TechnicalFeatureExtractor(BaseFeatureExtractor):
    """
    技术指标特征提取器

    提取所有核心技术指标的特征：
    1. 趋势特征（MA系列）
    2. 动能特征（MACD、动量）
    3. 振荡特征（RSI、KDJ）
    4. 波动率特征（布林带、ATR）
    5. 成交量特征
    """

    def __init__(
        self,
        # MA 参数
        ma_periods: List[int] = None,
        # MACD 参数
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # RSI 参数
        rsi_period: int = 14,
        # 布林带参数
        bb_period: int = 20,
        bb_std: float = 2.0,
        # ATR 参数
        atr_period: int = 14,
        # 动量参数
        momentum_period: int = 10,
        # KDJ 参数
        kdj_n: int = 9,
        kdj_m1: int = 3,
        kdj_m2: int = 3,
    ):
        """
        初始化技术指标特征提取器

        Args:
            ma_periods: MA周期列表，默认 [5, 10, 20, 60]
            macd_fast: MACD快线周期
            macd_slow: MACD慢线周期
            macd_signal: MACD信号线周期
            rsi_period: RSI周期
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            atr_period: ATR周期
            momentum_period: 动量周期
            kdj_n: KDJ的N参数
            kdj_m1: KDJ的M1参数
            kdj_m2: KDJ的M2参数
        """
        super().__init__(name="technical")

        # 设置默认参数
        if ma_periods is None:
            ma_periods = [5, 10, 20, 60]

        self.ma_periods = ma_periods
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.momentum_period = momentum_period
        self.kdj_n = kdj_n
        self.kdj_m1 = kdj_m1
        self.kdj_m2 = kdj_m2

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有技术指标特征

        Args:
            df: 原始OHLCV数据

        Returns:
            包含所有特征的DataFrame
        """
        self._validate_data(df)

        result = df.copy()

        # 1. 趋势特征
        result = self._extract_ma_features(result)
        result = self._extract_trend_features(result)

        # 2. 动能特征
        result = self._extract_macd_features(result)
        result = self._extract_momentum_features(result)

        # 3. 振荡特征
        result = self._extract_rsi_features(result)
        result = self._extract_kdj_features(result)

        # 4. 波动率特征
        result = self._extract_bollinger_bands_features(result)
        result = self._extract_atr_features(result)

        # 5. 成交量特征
        result = self._extract_volume_features(result)

        return result

    def _extract_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取MA相关特征

        特征：
        - MA{period}: 移动平均线值
        - MA{period}_ratio: 价格与MA的比率
        - MA{period}_slope: MA斜率
        """
        ma_indicator = MA()

        for period in self.ma_periods:
            # 计算MA
            df = ma_indicator.calculate(df, period=period)

            # 价格与MA的比率（衡量价格相对MA的位置）
            ma_col = f"MA{period}"
            df[f"MA{period}_ratio"] = df["close"] / df[ma_col] - 1

            # MA斜率（衡量趋势强度）
            df[f"MA{period}_slope"] = df[ma_col].diff(1) / df[ma_col].shift(1)

            # 注册特征名
            self._register_feature(f"MA{period}")
            self._register_feature(f"MA{period}_ratio")
            self._register_feature(f"MA{period}_slope")

        return df

    def _extract_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取趋势特征

        特征：
        - bullish_alignment: 多头排列（短>中>长）
        - ma_cross_signal: MA交叉信号
        """
        if len(self.ma_periods) >= 3:
            ma_fast = f"MA{self.ma_periods[0]}"
            ma_mid = f"MA{self.ma_periods[1]}"
            ma_slow = f"MA{self.ma_periods[2]}"

            # 多头排列
            df["bullish_alignment"] = (
                (df[ma_fast] > df[ma_mid])
                & (df[ma_mid] > df[ma_slow])
                & (df[ma_fast] > df[ma_fast].shift(1))
            ).astype(int)

            # 空头排列
            df["bearish_alignment"] = (
                (df[ma_fast] < df[ma_mid])
                & (df[ma_mid] < df[ma_slow])
                & (df[ma_fast] < df[ma_fast].shift(1))
            ).astype(int)

            self._register_feature("bullish_alignment")
            self._register_feature("bearish_alignment")

        return df

    def _extract_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取MACD相关特征

        特征：
        - DIF: 快线
        - DEA: 慢线（信号线）
        - MACD: 柱状图
        - MACD_zero_above: 是否在零轴上方
        - MACD_slope: MACD柱状图斜率
        """
        # 计算MACD
        macd_indicator = MACD()
        df = macd_indicator.calculate(
            df,
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal,
        )

        # 零轴上方（趋势强度）
        df["MACD_zero_above"] = (df["DIF"] > 0).astype(int)

        # MACD柱状图斜率（动能变化）
        df["MACD_slope"] = df["MACD"].diff(1)

        # DIF与DEA的距离（趋势强度）
        df["MACD_distance"] = df["DIF"] - df["DEA"]

        self._register_feature("DIF")
        self._register_feature("DEA")
        self._register_feature("MACD")
        self._register_feature("MACD_zero_above")
        self._register_feature("MACD_slope")
        self._register_feature("MACD_distance")

        return df

    def _extract_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取动量特征

        特征：
        - ROC: 变化率
        - MOM: 动量值
        - acceleration: 价格加速度（二阶差分）
        """
        # ROC (Rate of Change)
        df[f"ROC_{self.momentum_period}"] = (
            df["close"].pct_change(self.momentum_period) * 100
        )

        # MOM (Momentum)
        df[f"MOM_{self.momentum_period}"] = (
            df["close"] - df["close"].shift(self.momentum_period)
        )

        # 价格加速度（二阶差分）
        df["price_acceleration"] = df["close"].diff(1).diff(1)

        self._register_feature(f"ROC_{self.momentum_period}")
        self._register_feature(f"MOM_{self.momentum_period}")
        self._register_feature("price_acceleration")

        return df

    def _extract_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取RSI相关特征

        特征：
        - RSI: RSI值
        - RI_overbought: 是否超买
        - RSI_oversold: 是否超卖
        - RSI_slope: RSI斜率
        """
        # 计算RSI
        rsi_indicator = RSI()
        df = rsi_indicator.calculate(df, period=self.rsi_period)

        # RSI斜率
        df["RSI_slope"] = df["RSI"].diff(1)

        # 超买超卖区域
        df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
        df["RSI_oversold"] = (df["RSI"] < 30).astype(int)

        self._register_feature("RSI")
        self._register_feature("RSI_slope")
        self._register_feature("RSI_overbought")
        self._register_feature("RSI_oversold")

        return df

    def _extract_kdj_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取KDJ相关特征

        特征：
        - K: K值
        - D: D值
        - J: J值
        - KDJ_golden_cross: 金叉
        - KDJ_death_cross: 死叉
        """
        # 计算RSV
        low_min = df["low"].rolling(window=self.kdj_n).min()
        high_max = df["high"].rolling(window=self.kdj_n).max()
        df["RSV"] = (df["close"] - low_min) / (high_max - low_min) * 100

        # 初始化K、D、J
        df["K"] = 50.0
        df["D"] = 50.0

        # 计算K、D（平滑处理）- 使用iloc避免索引问题
        for i in range(1, len(df)):
            df.iloc[i, df.columns.get_loc("K")] = (2 / 3) * df.iloc[i - 1]["K"] + (1 / 3) * df.iloc[i]["RSV"]
            df.iloc[i, df.columns.get_loc("D")] = (2 / 3) * df.iloc[i - 1]["D"] + (1 / 3) * df.iloc[i]["K"]

        # 计算J
        df["J"] = 3 * df["K"] - 2 * df["D"]

        # 金叉死叉
        df["KDJ_golden_cross"] = ((df["K"] > df["D"]) & (df["K"].shift(1) <= df["D"].shift(1))).astype(int)
        df["KDJ_death_cross"] = ((df["K"] < df["D"]) & (df["K"].shift(1) >= df["D"].shift(1))).astype(int)

        self._register_feature("K")
        self._register_feature("D")
        self._register_feature("J")
        self._register_feature("KDJ_golden_cross")
        self._register_feature("KDJ_death_cross")

        return df

    def _extract_bollinger_bands_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取布林带相关特征

        特征：
        - BB_upper: 上轨
        - BB_lower: 下轨
        - BB_middle: 中轨（MA）
        - BB_width: 带宽（波动率）
        - BB_position: 价格在布林带中的位置
        - BB_squeeze: 收缩（低波动率）
        """
        # 计算中轨（MA）
        df["BB_middle"] = df["close"].rolling(window=self.bb_period).mean()

        # 计算标准差
        std = df["close"].rolling(window=self.bb_period).std()

        # 计算上下轨
        df["BB_upper"] = df["BB_middle"] + self.bb_std * std
        df["BB_lower"] = df["BB_middle"] - self.bb_std * std

        # 带宽（波动率指标）
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

        # 价格在布林带中的位置（0-1）
        df["BB_position"] = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

        # 收缩（带宽低于历史20%分位数）
        bandwidth_threshold = df["BB_width"].rolling(window=50).quantile(0.2)
        df["BB_squeeze"] = (df["BB_width"] < bandwidth_threshold).astype(int)

        self._register_feature("BB_upper")
        self._register_feature("BB_lower")
        self._register_feature("BB_middle")
        self._register_feature("BB_width")
        self._register_feature("BB_position")
        self._register_feature("BB_squeeze")

        return df

    def _extract_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取ATR相关特征

        特征：
        - ATR: 真实波动幅度
        - ATR_ratio: ATR相对价格
        """
        # 计算真实波动幅度
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR（移动平均）
        df["ATR"] = true_range.rolling(window=self.atr_period).mean()

        # ATR相对价格（归一化）
        df["ATR_ratio"] = df["ATR"] / df["close"]

        self._register_feature("ATR")
        self._register_feature("ATR_ratio")

        return df

    def _extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取成交量相关特征

        特征：
        - volume_ratio: 量比（当日成交量/5日均量）
        - volume_ma5: 5日平均成交量
        - volume_slope: 成交量斜率
        - price_volume_trend: 价量趋势
        """
        # 成交量MA
        df["volume_ma5"] = df["volume"].rolling(window=5).mean()

        # 量比
        df["volume_ratio"] = df["volume"] / df["volume_ma5"]

        # 成交量斜率
        df["volume_slope"] = df["volume"].diff(1) / df["volume"].shift(1)

        # 价量趋势（OBV简化版）
        price_change = df["close"].diff(1)
        volume_direction = np.sign(price_change)
        df["price_volume_trend"] = (volume_direction * df["volume"]).cumsum()

        self._register_feature("volume_ma5")
        self._register_feature("volume_ratio")
        self._register_feature("volume_slope")
        self._register_feature("price_volume_trend")

        return df
