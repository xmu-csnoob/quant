"""
交易策略模块

包含各类交易策略的实现：
- BaseStrategy: 策略抽象基类
- Signal: 交易信号
- Position: 持仓信息
- MaMacdRsiStrategy: MA + MACD + RSI 组合策略

策略类型：
- TREND_FOLLOWING: 趋势跟踪
- MEAN_REVERSION: 均值回归
- MOMENTUM: 动量
- BREAKOUT: 突破
- ARBITRAGE: 套利
- STATISTICAL: 统计套利
- MARKET_MAKING: 做市
"""

from .base import (
    # 枚举类型
    StrategyType,
    AssetClass,
    Frequency,
    SignalType,
    PositionType,
    # 数据类
    BaseStrategy,
    Signal,
    Position,
)
from .ma_macd_rsi import MaMacdRsiStrategy

__all__ = [
    # 枚举
    "StrategyType",
    "AssetClass",
    "Frequency",
    "SignalType",
    "PositionType",
    # 核心类
    "BaseStrategy",
    "Signal",
    "Position",
    # 具体策略
    "MaMacdRsiStrategy",
]
