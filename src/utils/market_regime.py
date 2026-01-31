"""
市场环境识别器

识别当前市场处于牛市、熊市还是震荡市
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Optional
from loguru import logger


class MarketRegime(Enum):
    """市场环境"""
    BULL = "牛市"      # 上涨趋势
    BEAR = "熊市"      # 下跌趋势
    SIDEWAYS = "震荡市"  # 横盘震荡
    VOLATILE = "高波动"  # 高波动状态


class MarketRegimeDetector:
    """
    市场环境识别器

    方法：
    1. 趋势识别：MA斜率、价格相对MA位置
    2. 波动率识别：ATR、历史波动率
    3. 动量识别：累计收益率
    """

    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 60,
        lookback: int = 60,
    ):
        """
        初始化

        Args:
            ma_short: 短期均线周期
            ma_long: 长期均线周期
            lookback: 回看天数
        """
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, index: int = None) -> MarketRegime:
        """
        检测市场环境

        Args:
            df: OHLCV数据
            index: 检测位置的索引（None=最新位置）

        Returns:
            市场环境
        """
        if index is None:
            index = len(df) - 1

        if index < self.ma_long:
            return MarketRegime.SIDEWAYS

        # 获取回看窗口
        start_idx = max(0, index - self.lookback)
        df_window = df.iloc[start_idx:index+1].copy()

        # 1. 计算各项指标
        ma_short = df_window["close"].rolling(self.ma_short).mean()
        ma_long = df_window["close"].rolling(self.ma_long).mean()

        # 趋势特征
        trend_slope = (ma_short.iloc[-1] - ma_short.iloc[-self.ma_short]) / ma_short.iloc[-self.ma_short]
        ma_position = (df_window["close"].iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]

        # 动量特征
        momentum = (df_window["close"].iloc[-1] / df_window["close"].iloc[0] - 1)

        # 波动率特征
        returns = df_window["close"].pct_change()
        volatility = returns.std()

        # 2. 综合判断
        regime = self._classify_regime(
            trend_slope=trend_slope,
            ma_position=ma_position,
            momentum=momentum,
            volatility=volatility,
        )

        return regime

    def _classify_regime(
        self,
        trend_slope: float,
        ma_position: float,
        momentum: float,
        volatility: float,
    ) -> MarketRegime:
        """
        根据特征分类市场环境

        规则：
        - 牛市：趋势向上、价格在MA上方、动量为正
        - 熊市：趋势向下、价格在MA下方、动量为负
        - 高波动：波动率异常高
        - 震荡市：其他情况
        """
        # 判断波动率（日波动率 > 3% 为高波动）
        is_volatile = volatility > 0.03

        if is_volatile:
            return MarketRegime.VOLATILE

        # 判断趋势
        is_uptrend = (
            trend_slope > 0.001 and  # MA向上倾斜
            ma_position > 0.02 and   # 价格高于MA 2%
            momentum > 0.05          # 60日涨幅 > 5%
        )

        is_downtrend = (
            trend_slope < -0.001 and  # MA向下倾斜
            ma_position < -0.02 and   # 价格低于MA 2%
            momentum < -0.05          # 60日跌幅 > 5%
        )

        if is_uptrend:
            return MarketRegime.BULL
        elif is_downtrend:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def detect_with_confidence(
        self,
        df: pd.DataFrame,
        index: int = None
    ) -> tuple[MarketRegime, float]:
        """
        检测市场环境并返回置信度

        Returns:
            (市场环境, 置信度 0-1)
        """
        if index is None:
            index = len(df) - 1

        if index < self.ma_long:
            return MarketRegime.SIDEWAYS, 0.5

        start_idx = max(0, index - self.lookback)
        df_window = df.iloc[start_idx:index+1].copy()

        ma_short = df_window["close"].rolling(self.ma_short).mean()
        ma_long = df_window["close"].rolling(self.ma_long).mean()

        trend_slope = (ma_short.iloc[-1] - ma_short.iloc[-self.ma_short]) / ma_short.iloc[-self.ma_short]
        ma_position = (df_window["close"].iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        momentum = (df_window["close"].iloc[-1] / df_window["close"].iloc[0] - 1)

        # 计算置信度（基于信号强度）
        regime_signals = {
            MarketRegime.BULL: 0,
            MarketRegime.BEAR: 0,
            MarketRegime.SIDEWAYS: 0,
        }

        # 趋势信号
        if trend_slope > 0.002:
            regime_signals[MarketRegime.BULL] += 1
        elif trend_slope < -0.002:
            regime_signals[MarketRegime.BEAR] += 1
        else:
            regime_signals[MarketRegime.SIDEWAYS] += 1

        # MA位置信号
        if ma_position > 0.03:
            regime_signals[MarketRegime.BULL] += 1
        elif ma_position < -0.03:
            regime_signals[MarketRegime.BEAR] += 1
        else:
            regime_signals[MarketRegime.SIDEWAYS] += 1

        # 动量信号
        if momentum > 0.08:
            regime_signals[MarketRegime.BULL] += 1
        elif momentum < -0.08:
            regime_signals[MarketRegime.BEAR] += 1
        else:
            regime_signals[MarketRegime.SIDEWAYS] += 1

        # 找出最强的信号
        regime = max(regime_signals.keys(), key=lambda k: regime_signals[k])
        confidence = regime_signals[regime] / 3.0

        return regime, confidence

    def get_regime_history(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        获取完整的市场环境历史

        Returns:
            添加了 regime 和 regime_confidence 列的DataFrame
        """
        df = df.copy()

        regimes = []
        confidences = []

        for i in range(len(df)):
            regime, conf = self.detect_with_confidence(df, i)
            regimes.append(regime.value)
            confidences.append(conf)

        df["market_regime"] = regimes
        df["regime_confidence"] = confidences

        return df
