"""
优化版机器学习特征提取器

移除了时间特征和低价值特征，专注于有预测价值的技术特征

特征数量：~45个（原58个）

移除的特征：
1. 时间特征（6个）- 与收益相关性≈0，易过拟合
   - f_day_of_week, f_month, f_quarter
   - f_month_start, f_month_end, f_day_of_year

2. 低价值统计特征（2个）- 重要性最低
   - f_return_skew_20, f_return_kurt_20

3. 低价值成交量特征（3个）
   - f_obv_ratio, f_obv_ma, f_upper_shadow_ratio

保留的核心特征：
- 价格动量：return_1d, return_3d, return_5d, return_10d, momentum_5_20
- 技术指标：rsi, macd, macd_hist, bb_position, atr, atr_ratio
- 均线系统：ma_5/10/20/60_ratio, ma20_slope, ma_alignment
- 成交量：volume_ratio, price_volume_trend, obv
- 波动率：volatility_10/20, volatility_change
- 交互特征：rsi_trend, vol_momentum, vol_price, atr_rsi
- 价格模式：gap, daily_range, lower_shadow_ratio, consecutive_up/down
- 成交量模式：volume_ratio_5/20/60, volume_change, divergence, amount_ratio
- 统计特征：return_std_10, price_percentile_20, yesterday_return
          positive_ratio_20, cum_return_5/10/20
"""

import pandas as pd
import numpy as np
from typing import List
from loguru import logger


class OptimizedFeatureExtractor:
    """
    优化版特征提取器

    专注于有经济学逻辑支撑的技术特征，避免过拟合
    """

    def __init__(self, prediction_period: int = 5):
        self.prediction_period = prediction_period

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取优化特征

        Args:
            df: 原始OHLCV数据，需包含列：
                - open, high, low, close, volume, amount
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

        # 7. 价格模式特征（选择性保留）
        df = self._add_price_pattern_features(df)

        # 8. 成交量模式特征
        df = self._add_volume_pattern_features(df)

        # 9. 统计特征（选择性保留）
        df = self._add_statistical_features(df)

        feature_count = len([c for c in df.columns if c.startswith("f_")])
        logger.info(f"提取了 {feature_count} 个优化特征")

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格动量特征（使用昨日收盘价避免前视偏差）"""
        close_yesterday = df["close"].shift(1)

        # 各种周期的收益率
        for period in [1, 3, 5, 10]:
            df[f"f_return_{period}d"] = close_yesterday.pct_change(period)

        # 动量强弱
        df["f_momentum_5_20"] = (
            close_yesterday.pct_change(5) - close_yesterday.pct_change(20)
        )

        return df

    def _add_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """技术指标特征（使用昨日收盘价避免前视偏差）"""
        close_yesterday = df["close"].shift(1)

        # RSI (14日)
        df["f_rsi"] = self._calculate_rsi(close_yesterday, 14)

        # MACD
        macd_line = close_yesterday.ewm(span=12).mean() - close_yesterday.ewm(span=26).mean()
        signal_line = macd_line.ewm(span=9).mean()
        df["f_macd"] = macd_line - signal_line
        df["f_macd_hist"] = macd_line - signal_line

        # 布林带位置
        bb_mid = close_yesterday.rolling(20).mean()
        bb_std = close_yesterday.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = bb_upper - bb_lower
        df["f_bb_position"] = np.divide(
            close_yesterday - bb_lower, bb_width,
            where=bb_width != 0,
            out=np.full_like(close_yesterday, 0.5, dtype=float)
        )

        # ATR (14日)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["f_atr"] = tr.rolling(14).mean()
        df["f_atr_ratio"] = np.divide(
            df["f_atr"], close_yesterday,
            where=close_yesterday != 0,
            out=np.zeros_like(close_yesterday, dtype=float)
        )

        return df

    def _add_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """均线系统特征（使用昨日收盘价避免前视偏差）"""
        close_yesterday = df["close"].shift(1)

        # 价格相对均线的位置
        for period in [5, 10, 20, 60]:
            ma = close_yesterday.rolling(period).mean()
            df[f"f_ma_{period}_ratio"] = close_yesterday / ma - 1

        # 均线斜率
        ma20 = close_yesterday.rolling(20).mean()
        df["f_ma20_slope"] = ma20.pct_change(5)

        # 多头排列（短期>长期）
        ma5 = close_yesterday.rolling(5).mean()
        ma20 = close_yesterday.rolling(20).mean()
        ma60 = close_yesterday.rolling(60).mean()
        df["f_ma_alignment"] = ((ma5 > ma20) & (ma20 > ma60)).astype(int)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特征（修复除零风险）"""
        # 量比
        vol_ma5 = df["volume"].rolling(5).mean()
        df["f_volume_ratio"] = np.divide(
            df["volume"], vol_ma5,
            where=vol_ma5 != 0,
            out=np.ones_like(df["volume"], dtype=float)
        )

        # 量价趋势
        price_change = df["close"].pct_change()
        volume_change = df["volume"].pct_change()
        df["f_price_volume_trend"] = price_change * volume_change

        # OBV（保留核心，移除衍生特征）
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["f_obv"] = obv
        # 移除 f_obv_ma 和 f_obv_ratio（低价值）

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率特征"""
        returns = df["close"].pct_change()
        df["f_volatility_10"] = returns.rolling(10).std()
        df["f_volatility_20"] = returns.rolling(20).std()
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

    def _add_price_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格模式特征（选择性保留）"""
        # 跳空
        df["f_gap"] = (df["open"] / df["close"].shift(1) - 1).fillna(0)

        # 振幅
        df["f_daily_range"] = (df["high"] - df["low"]) / df["open"]

        # 影线比例（只保留下影线，移除上影线）
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        daily_range = df["high"] - df["low"]
        df["f_lower_shadow_ratio"] = np.divide(
            lower_shadow, daily_range,
            where=daily_range != 0,
            out=np.zeros_like(lower_shadow, dtype=float)
        )
        # 移除 f_upper_shadow_ratio（低价值）

        # 连续上涨/下跌天数
        price_change = np.sign(df["close"].diff())
        df["f_consecutive_up"] = (price_change > 0).astype(int).rolling(5).sum()
        df["f_consecutive_down"] = (price_change < 0).astype(int).rolling(5).sum()

        # 收益标准差
        df["f_return_std_10"] = df["close"].pct_change().rolling(10).std()

        # 价格位置
        df["f_price_percentile_20"] = df["close"].shift(1).rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        )

        # 前日涨跌
        df["f_yesterday_return"] = df["close"].pct_change(1)

        return df

    def _add_volume_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量模式特征"""
        # 多周期量比
        for period in [5, 20, 60]:
            vol_ma = df["volume"].rolling(period).mean()
            df[f"f_volume_ratio_{period}"] = np.divide(
                df["volume"], vol_ma,
                where=vol_ma != 0,
                out=np.ones_like(df["volume"], dtype=float)
            )

        # 移除 f_volume_surge, f_volume_shrink（低价值二元特征）

        # 成交量变化率
        df["f_volume_change"] = df["volume"].pct_change(5)

        # 量价背离
        price_up = df["close"] > df["close"].shift(5)
        vol_down = df["volume"] < df["volume"].shift(5)
        df["f_divergence"] = (price_up & vol_down).astype(int)

        # 成交额占比
        df["f_amount_ratio"] = df["amount"] / (df["open"] * df["volume"])

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计特征（选择性保留）"""
        returns = df["close"].pct_change()

        # 移除 f_return_skew_20, f_return_kurt_20（低价值）

        # 正收益比例
        df["f_positive_ratio_20"] = (returns > 0).rolling(20).mean()

        # 累计收益
        for period in [5, 10, 20]:
            df[f"f_cum_return_{period}"] = returns.rolling(period).apply(
                lambda x: (1 + x).prod() - 1
            )

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = np.divide(gain, loss, where=loss != 0, out=np.full_like(gain, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100.0)
        return rsi

    def get_feature_names(self) -> List[str]:
        """获取所有特征名"""
        features = []

        # 动量
        for p in [1, 3, 5, 10]:
            features.append(f"f_return_{p}d")
        features.append("f_momentum_5_20")

        # 技术指标
        features.extend([
            "f_rsi", "f_macd", "f_macd_hist", "f_bb_position",
            "f_atr", "f_atr_ratio",
        ])

        # 均线
        for p in [5, 10, 20, 60]:
            features.append(f"f_ma_{p}_ratio")
        features.extend(["f_ma20_slope", "f_ma_alignment"])

        # 成交量
        features.extend([
            "f_volume_ratio", "f_price_volume_trend", "f_obv",
        ])

        # 波动率
        features.extend([
            "f_volatility_10", "f_volatility_20", "f_volatility_change",
        ])

        # 交互
        features.extend([
            "f_rsi_trend", "f_vol_momentum", "f_vol_price", "f_atr_rsi",
        ])

        # 价格模式
        features.extend([
            "f_gap", "f_daily_range", "f_lower_shadow_ratio",
            "f_consecutive_up", "f_consecutive_down",
            "f_return_std_10", "f_price_percentile_20", "f_yesterday_return",
        ])

        # 成交量模式
        for p in [5, 20, 60]:
            features.append(f"f_volume_ratio_{p}")
        features.extend([
            "f_volume_change", "f_divergence", "f_amount_ratio",
        ])

        # 统计
        features.extend([
            "f_positive_ratio_20",
            "f_cum_return_5", "f_cum_return_10", "f_cum_return_20",
        ])

        return features
