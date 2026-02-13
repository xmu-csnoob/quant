"""
Strategy and backtest related schemas.
策略和回测相关数据模型
"""

from enum import Enum
from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, Field


class StrategyStatus(str, Enum):
    """策略状态"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class SignalDirection(str, Enum):
    """信号方向"""
    BUY = "buy"
    SELL = "sell"


class BacktestStatus(str, Enum):
    """回测状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== 策略相关 ====================

class Strategy(BaseModel):
    """策略信息"""
    id: str = Field(description="策略ID")
    name: str = Field(description="策略名称")
    description: str = Field(description="策略描述")
    status: StrategyStatus = Field(description="策略状态")
    return_rate: float = Field(description="收益率(%)")
    win_rate: float = Field(description="胜率(%)")
    trade_count: int = Field(description="交易次数")


class Signal(BaseModel):
    """交易信号"""
    id: str = Field(description="信号ID")
    strategy_id: str = Field(description="策略ID")
    strategy_name: str = Field(description="策略名称")
    code: str = Field(description="股票代码")
    name: str = Field(description="股票名称")
    direction: SignalDirection = Field(description="信号方向")
    price: float = Field(description="信号价格")
    confidence: float = Field(description="置信度(%)")
    created_at: datetime = Field(description="生成时间")


# ==================== 回测相关 ====================

class BacktestConfig(BaseModel):
    """回测配置"""
    strategy_id: str = Field(description="策略ID")
    start_date: date = Field(description="开始日期")
    end_date: date = Field(description="结束日期")
    initial_capital: float = Field(default=1000000, description="初始资金")
    commission_rate: float = Field(default=0.0003, description="手续费率")
    slippage_rate: float = Field(default=0.001, description="滑点率")


class EquityPoint(BaseModel):
    """净值曲线点"""
    date: str = Field(description="日期")
    equity: float = Field(description="净值")
    return_rate: float = Field(description="收益率(%)")


class Trade(BaseModel):
    """交易记录"""
    date: str = Field(description="交易日期")
    code: str = Field(description="股票代码")
    name: str = Field(description="股票名称")
    direction: SignalDirection = Field(description="买卖方向")
    price: float = Field(description="成交价格")
    shares: int = Field(description="成交数量")
    profit: float = Field(description="盈亏")


class BacktestResult(BaseModel):
    """回测结果"""
    id: str = Field(description="回测ID")
    status: BacktestStatus = Field(description="回测状态")
    config: BacktestConfig = Field(description="回测配置")

    # 统计指标
    total_return: float = Field(default=0, description="总收益率(%)")
    annual_return: float = Field(default=0, description="年化收益率(%)")
    max_drawdown: float = Field(default=0, description="最大回撤(%)")
    sharpe_ratio: float = Field(default=0, description="夏普比率")
    win_rate: float = Field(default=0, description="胜率(%)")
    profit_factor: float = Field(default=0, description="盈亏比")
    trade_count: int = Field(default=0, description="交易次数")

    # T+1统计
    t1_violations: int = Field(default=0, description="T+1违规尝试次数")
    t1_skipped_sells: int = Field(default=0, description="因T+1跳过的卖出")

    # 时间序列数据
    equity_curve: list[EquityPoint] = Field(default_factory=list, description="净值曲线")
    trades: list[Trade] = Field(default_factory=list, description="交易记录")


# ==================== 数据相关 ====================

class DataStatus(BaseModel):
    """数据状态"""
    total_stocks: int = Field(description="股票总数")
    last_update: datetime = Field(description="最后更新时间")
    data_sources: list[str] = Field(description="数据源")
    update_status: str = Field(description="更新状态")


class KlineData(BaseModel):
    """K线数据"""
    date: str = Field(description="日期")
    open: float = Field(description="开盘价")
    high: float = Field(description="最高价")
    low: float = Field(description="最低价")
    close: float = Field(description="收盘价")
    volume: float = Field(description="成交量")
    amount: float = Field(description="成交额")
