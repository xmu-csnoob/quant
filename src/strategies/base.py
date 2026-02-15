"""
策略基类模块

定义交易策略的抽象接口和基础数据结构

设计原则：
1. 策略只负责生成信号，不管回测和执行
2. 回测引擎独立于策略实现
3. 策略按类型分类，不按功能（选股/择时/对冲）分类
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional, ClassVar
import pandas as pd
import numpy as np


class StrategyType(Enum):
    """策略类型（按策略逻辑分类）"""

    TREND_FOLLOWING = "trend_following"  # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"  # 均值回归
    MOMENTUM = "momentum"  # 动量
    BREAKOUT = "breakout"  # 突破
    ARBITRAGE = "arbitrage"  # 套利
    STATISTICAL = "statistical"  # 统计套利
    MARKET_MAKING = "market_making"  # 做市


class AssetClass(Enum):
    """资产类别"""

    STOCK = "stock"  # 股票
    FUTURES = "futures"  # 期货
    OPTIONS = "options"  # 期权
    FOREX = "forex"  # 外汇
    CRYPTO = "crypto"  # 加密货币
    MULTI_ASSET = "multi_asset"  # 多资产


class Frequency(Enum):
    """交易频率"""

    TICK = "tick"  # 逐笔
    MINUTE_1 = "1m"  # 1分钟
    MINUTE_5 = "5m"  # 5分钟
    HOUR_1 = "1h"  # 1小时
    DAILY = "1d"  # 日线
    WEEKLY = "1w"  # 周线


class SignalType(Enum):
    """信号类型"""

    BUY = "buy"  # 买入信号
    SELL = "sell"  # 卖出信号
    HOLD = "hold"  # 持有/观望
    CLOSE = "close"  # 平仓


@dataclass
class Signal:
    """
    交易信号

    Attributes:
        date: 信号日期
        signal_type: 信号类型
        price: 信号价格
        symbol: 股票代码（可选，多标的交易时必须）
        reason: 信号原因（可选，用于调试）
        confidence: 信号置信度 (0-1)，可选
        quantity: 建议的交易数量（可选），如果不指定则由回测器计算
    """

    date: str
    signal_type: SignalType
    price: float
    symbol: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    quantity: Optional[int] = None

    def __str__(self) -> str:
        conf_str = f", confidence={self.confidence:.2f}" if self.confidence else ""
        symbol_str = f", symbol={self.symbol}" if self.symbol else ""
        return f"Signal({self.date}{symbol_str}, {self.signal_type.value}, price={self.price:.2f}{conf_str}, reason={self.reason})"


class PositionType(Enum):
    """持仓类型"""

    LONG = "long"  # 多头持仓
    SHORT = "short"  # 空头持仓
    FLAT = "flat"  # 空仓


@dataclass
class Position:
    """
    持仓信息（策略层）

    注意：系统中存在多个Position定义：
    - src.strategies.base.Position: 策略层，包含position_type等策略相关字段
    - src.trading.api.Position: 交易API层，包含available_quantity等交易相关字段
    - src.risk.manager.Position: 风控层，包含unrealized_pnl等风控相关字段

    TODO: 统一Position模型，避免数据传递时的字段丢失或误解

    Attributes:
        entry_date: 建仓日期
        entry_price: 建仓价格
        quantity: 持仓数量
        position_type: 持仓类型
        stop_loss: 止损价格（可选）
        take_profit: 止盈价格（可选）
    """

    entry_date: str
    entry_price: float
    quantity: int
    position_type: PositionType
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def is_long(self) -> bool:
        """是否多头持仓"""
        return self.position_type == PositionType.LONG

    @property
    def is_short(self) -> bool:
        """是否空头持仓"""
        return self.position_type == PositionType.SHORT

    @property
    def is_flat(self) -> bool:
        """是否空仓"""
        return self.position_type == PositionType.FLAT

    def current_value(self, current_price: float) -> float:
        """
        计算当前持仓市值

        Args:
            current_price: 当前价格

        Returns:
            当前市值
        """
        if self.is_flat:
            return 0.0

        if self.is_long:
            return self.quantity * current_price
        else:  # short
            return self.quantity * (2 * self.entry_price - current_price)

    def unrealized_pnl(self, current_price: float) -> float:
        """
        计算未实现盈亏

        Args:
            current_price: 当前价格

        Returns:
            未实现盈亏
        """
        if self.is_flat:
            return 0.0

        if self.is_long:
            return self.quantity * (current_price - self.entry_price)
        else:  # short
            return self.quantity * (self.entry_price - current_price)


class StrategyError(Exception):
    """策略异常"""

    pass


class BaseStrategy(ABC):
    """
    策略抽象基类

    所有交易策略都应继承此类并实现 generate_signals 方法

    设计原则：
    - 策略只负责生成交易信号
    - 不管回测、执行、风险控制
    - 这些由回测引擎和交易系统负责
    """

    # 策略元数据（子类覆盖）
    strategy_type: ClassVar[StrategyType] = StrategyType.TREND_FOLLOWING
    asset_class: ClassVar[AssetClass] = AssetClass.STOCK
    frequency: ClassVar[Frequency] = Frequency.DAILY

    def __init__(self, name: str = None):
        """
        初始化策略

        Args:
            name: 策略名称
        """
        self.name = name or self.__class__.__name__
        self._signals = []
        self._position = None

    def get_metadata(self) -> dict:
        """
        获取策略元数据

        Returns:
            策略元数据字典
        """
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "asset_class": self.asset_class.value,
            "frequency": self.frequency.value,
        }

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        """
        生成交易信号

        Args:
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            信号列表
        """
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略所需的技术指标

        子类可以重写此方法来计算自定义指标

        Args:
            df: 原始数据

        Returns:
            添加了指标列的 DataFrame
        """
        return df

    def get_signals(self) -> list[Signal]:
        """
        获取生成的所有信号

        Returns:
            信号列表
        """
        return self._signals

    def get_latest_signal(self) -> Optional[Signal]:
        """
        获取最新信号

        Returns:
            最新信号，如果没有则返回 None
        """
        return self._signals[-1] if self._signals else None
