"""
简单回测引擎

用于快速验证策略思路，不涉及复杂的风控和订单管理
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, SignalType, Position, PositionType


@dataclass
class Trade:
    """
    交易记录

    Attributes:
        entry_date: 建仓日期
        exit_date: 平仓日期
        entry_price: 建仓价格
        exit_price: 平仓价格
        quantity: 交易数量
        trade_type: 交易类型
        pnl: 盈亏
        pnl_ratio: 盈亏比例
        reason_exit: 卖出原因
    """

    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    trade_type: PositionType
    pnl: float
    pnl_ratio: float
    reason_exit: str

    def __str__(self) -> str:
        return (
            f"Trade({self.entry_date} -> {self.exit_date}, "
            f"{self.entry_price:.2f} -> {self.exit_price:.2f}, "
            f"PNL={self.pnl:.2f} ({self.pnl_ratio*100:.2f}%))"
        )


@dataclass
class BacktestResult:
    """
    回测结果

    Attributes:
        strategy_name: 策略名称
        start_date: 回测开始日期
        end_date: 回测结束日期
        initial_capital: 初始资金
        final_capital: 最终资金
        total_return: 总收益率
        trades: 交易记录列表
        trade_count: 交易次数
        win_count: 盈利次数
        lose_count: 亏损次数
        win_rate: 胜率
        max_drawdown: 最大回撤
        sharpe_ratio: 夏普比率
    """

    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    trades: List[Trade] = field(default_factory=list)
    trade_count: int = 0
    win_count: int = 0
    lose_count: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    def print_summary(self):
        """打印回测结果摘要"""
        print("=" * 70)
        print(f"回测报告: {self.strategy_name}")
        print("=" * 70)
        print(f"回测期间: {self.start_date} ~ {self.end_date}")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"最终资金: {self.final_capital:,.2f}")
        print(f"总收益率: {self.total_return*100:.2f}%")
        print("-" * 70)
        print(f"交易次数: {self.trade_count}")
        print(f"盈利次数: {self.win_count}")
        print(f"亏损次数: {self.lose_count}")
        print(f"胜率: {self.win_rate*100:.2f}%")
        print("-" * 70)
        print(f"最大回撤: {self.max_drawdown*100:.2f}%")
        print(f"夏普比率: {self.sharpe_ratio:.2f}")
        print("=" * 70)

    def print_trades(self, limit: int = 10):
        """打印交易记录"""
        print(f"\n最近 {min(limit, len(self.trades))} 笔交易:")
        for trade in self.trades[-limit:]:
            print(f"  {trade}")


class SimpleBacktester:
    """
    简单回测引擎

    特点：
    - 不考虑交易成本
    - 不考虑滑点
    - 假设每次交易固定数量
    - 适合快速验证策略思路
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital

    def run(self, strategy: BaseStrategy, df: pd.DataFrame) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略实例
            df: 包含 OHLCV 数据的 DataFrame

        Returns:
            回测结果
        """
        # 生成信号
        signals = strategy.generate_signals(df)

        if not signals:
            return BacktestResult(
                strategy_name=strategy.name,
                start_date=df.iloc[0]["trade_date"],
                end_date=df.iloc[-1]["trade_date"],
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0.0,
            )

        # 执行交易
        capital = self.initial_capital
        trades = []
        position = None
        max_capital = self.initial_capital
        max_drawdown = 0.0

        daily_returns = []  # 用于计算夏普比率

        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                if position is None:
                    # 买入
                    quantity = int(capital / signal.price / 100) * 100  # 整手
                    if quantity > 0:
                        position = Position(
                            entry_date=signal.date,
                            entry_price=signal.price,
                            quantity=quantity,
                            position_type=PositionType.LONG,
                        )
                        capital -= quantity * signal.price

            elif signal.signal_type == SignalType.SELL:
                if position is not None:
                    # 卖出
                    pnl = (signal.price - position.entry_price) * position.quantity
                    pnl_ratio = (signal.price - position.entry_price) / position.entry_price

                    capital += position.quantity * signal.price

                    trade = Trade(
                        entry_date=position.entry_date,
                        exit_date=signal.date,
                        entry_price=position.entry_price,
                        exit_price=signal.price,
                        quantity=position.quantity,
                        trade_type=position.position_type,
                        pnl=pnl,
                        pnl_ratio=pnl_ratio,
                        reason_exit=signal.reason or "",
                    )
                    trades.append(trade)
                    position = None

                    # 更新最大回撤
                    current_capital = capital
                    if current_capital > max_capital:
                        max_capital = current_capital
                    drawdown = (max_capital - current_capital) / max_capital
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

        # 计算统计指标
        total_return = (capital - self.initial_capital) / self.initial_capital

        win_count = sum(1 for t in trades if t.pnl > 0)
        lose_count = sum(1 for t in trades if t.pnl < 0)
        win_rate = win_count / len(trades) if trades else 0

        # 计算夏普比率（简化版）
        if trades:
            returns = [t.pnl_ratio for t in trades]
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            sharpe_ratio = sharpe * np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=df.iloc[0]["trade_date"],
            end_date=df.iloc[-1]["trade_date"],
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_return=total_return,
            trades=trades,
            trade_count=len(trades),
            win_count=win_count,
            lose_count=lose_count,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
        )
