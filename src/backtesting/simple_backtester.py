"""
简单回测引擎

用于快速验证策略思路，支持：
- 交易成本模拟（佣金、印花税、过户费）
- 滑点模拟
- T+1规则（A股）
- 基础绩效指标计算
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Dict
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from src.strategies.base import BaseStrategy, Signal, SignalType, Position, PositionType
from src.backtesting.costs import (
    CostConfig,
    CostBreakdown,
    TransactionCostCalculator,
    TradeSide,
)
from src.backtesting.slippage import (
    BaseSlippageModel,
    FixedSlippage,
    NoSlippage,
    SlippageResult,
)


# ==================== T+1规则辅助函数 ====================

def parse_date(date_str: str) -> date:
    """
    解析日期字符串

    支持格式：
    - YYYY-MM-DD
    - YYYYMMDD
    - datetime对象
    """
    if isinstance(date_str, date):
        return date_str
    if isinstance(date_str, datetime):
        return date_str.date()

    date_str = str(date_str)
    if '-' in date_str:
        return datetime.strptime(date_str[:10], '%Y-%m-%d').date()
    else:
        return datetime.strptime(date_str[:8], '%Y%m%d').date()


def get_next_trading_day(d: date) -> date:
    """
    获取下一交易日（简化版）

    注意：实际应使用交易日历，这里仅跳过周末
    """
    next_day = d + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day


def can_sell_after_t1(buy_date: date, sell_date: date) -> bool:
    """
    检查是否满足T+1规则

    A股T+1规则：
    - 当日买入的股票，下一交易日才能卖出
    - 如果买入日是周五，可卖日是下周一

    Args:
        buy_date: 买入日期
        sell_date: 卖出日期

    Returns:
        是否可以卖出
    """
    available_date = get_next_trading_day(buy_date)
    return sell_date >= available_date


@dataclass
class Trade:
    """
    交易记录

    Attributes:
        entry_date: 建仓日期
        exit_date: 平仓日期
        entry_price: 建仓价格（含滑点）
        exit_price: 平仓价格（含滑点）
        raw_entry_price: 原始建仓价格（不含滑点）
        raw_exit_price: 原始平仓价格（不含滑点）
        quantity: 交易数量
        trade_type: 交易类型
        pnl: 盈亏（扣除成本后）
        pnl_ratio: 盈亏比例（扣除成本后）
        gross_pnl: 毛利润（扣除成本前）
        total_cost: 总交易成本
        buy_cost: 买入成本明细
        sell_cost: 卖出成本明细
        buy_slippage: 买入滑点
        sell_slippage: 卖出滑点
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
    raw_entry_price: float = 0.0
    raw_exit_price: float = 0.0
    gross_pnl: float = 0.0
    total_cost: float = 0.0
    buy_cost: Optional[dict] = None
    sell_cost: Optional[dict] = None
    buy_slippage: float = 0.0
    sell_slippage: float = 0.0

    def __str__(self) -> str:
        cost_str = f", 成本={self.total_cost:.2f}" if self.total_cost > 0 else ""
        return (
            f"Trade({self.entry_date} -> {self.exit_date}, "
            f"{self.entry_price:.2f} -> {self.exit_price:.2f}, "
            f"PNL={self.pnl:.2f} ({self.pnl_ratio*100:.2f}%){cost_str})"
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
        total_costs: 总交易成本
        total_slippage: 总滑点金额
        cost_config: 成本配置
        t1_violations: T+1违规次数（尝试当天买入后卖出）
        t1_skipped_sells: 因T+1跳过的卖出信号数
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
    total_costs: float = 0.0
    total_slippage: float = 0.0
    cost_config: Optional[CostConfig] = None
    t1_violations: int = 0  # T+1违规次数
    t1_skipped_sells: int = 0  # 因T+1跳过的卖出

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
        if self.total_costs > 0:
            print("-" * 70)
            print(f"总交易成本: {self.total_costs:,.2f}")
            print(f"总滑点金额: {self.total_slippage:,.2f}")
        if self.t1_violations > 0 or self.t1_skipped_sells > 0:
            print("-" * 70)
            print(f"T+1违规尝试: {self.t1_violations}")
            print(f"T+1跳过卖出: {self.t1_skipped_sells}")
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
    - 支持交易成本模拟（佣金、印花税、过户费）
    - 支持滑点模拟
    - 支持T+1规则（A股）
    - 假设每次交易固定数量
    - 适合快速验证策略思路
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_config: Optional[CostConfig] = None,
        slippage_model: Optional[BaseSlippageModel] = None,
        enable_t1_rule: bool = True,  # 是否启用T+1规则
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            cost_config: 成本配置，默认使用A股标准费率
            slippage_model: 滑点模型，默认无滑点
            enable_t1_rule: 是否启用T+1规则，默认启用
        """
        self.initial_capital = initial_capital
        self.cost_config = cost_config or CostConfig.default()
        self.cost_calculator = TransactionCostCalculator(self.cost_config)
        self.slippage_model = slippage_model or NoSlippage()
        self.enable_t1_rule = enable_t1_rule

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
                start_date=str(df.iloc[0]["trade_date"]),
                end_date=str(df.iloc[-1]["trade_date"]),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0.0,
                cost_config=self.cost_config,
                t1_violations=0,
                t1_skipped_sells=0,
            )

        # 执行交易
        capital = Decimal(str(self.initial_capital))
        trades: List[Trade] = []
        position = None
        position_buy_date: Optional[date] = None  # 记录买入日期（用于T+1检查）
        max_capital = Decimal(str(self.initial_capital))
        max_drawdown = 0.0
        total_costs = Decimal("0")
        total_slippage = Decimal("0")
        t1_violations = 0  # T+1违规尝试次数
        t1_skipped_sells = 0  # 因T+1跳过的卖出信号数

        for signal in signals:
            raw_price = Decimal(str(signal.price))

            if signal.signal_type == SignalType.BUY:
                if position is None:
                    # 应用滑点
                    slippage_result = self.slippage_model.apply_slippage(
                        raw_price, TradeSide.BUY
                    )
                    adjusted_price = slippage_result.adjusted_price

                    # 计算买入数量
                    if signal.quantity is not None:
                        quantity = signal.quantity
                    else:
                        # 预留成本和滑点空间
                        effective_price = adjusted_price * (1 + self.cost_calculator.get_effective_buy_rate())
                        quantity = int(float(capital) / float(effective_price) / 100) * 100

                    if quantity > 0:
                        # 计算交易金额和成本
                        trade_amount = adjusted_price * quantity
                        cost_breakdown = self.cost_calculator.calculate_buy_cost(trade_amount)
                        total_cost = cost_breakdown.total

                        # 实际扣除资金
                        actual_cost = trade_amount + total_cost
                        if actual_cost <= capital:
                            position = Position(
                                entry_date=signal.date,
                                entry_price=float(adjusted_price),
                                quantity=quantity,
                                position_type=PositionType.LONG,
                            )
                            capital -= actual_cost
                            total_costs += total_cost
                            total_slippage += slippage_result.slippage_amount * quantity

                            # 保存买入信息用于后续计算
                            position._raw_entry_price = float(raw_price)
                            position._buy_cost = {
                                "commission": float(cost_breakdown.commission),
                                "stamp_duty": float(cost_breakdown.stamp_duty),
                                "transfer_fee": float(cost_breakdown.transfer_fee),
                                "total": float(total_cost),
                            }
                            position._buy_slippage = float(slippage_result.slippage_rate)

                            # 记录买入日期（用于T+1检查）
                            position_buy_date = parse_date(signal.date)

            elif signal.signal_type == SignalType.SELL:
                if position is not None:
                    # T+1检查：当天买入的不能卖出
                    sell_date = parse_date(signal.date)
                    if self.enable_t1_rule and position_buy_date is not None:
                        if not can_sell_after_t1(position_buy_date, sell_date):
                            # T+1违规：跳过此卖出信号
                            t1_violations += 1
                            t1_skipped_sells += 1
                            continue  # 跳过本次卖出

                    # 应用滑点
                    # 应用滑点
                    slippage_result = self.slippage_model.apply_slippage(
                        raw_price, TradeSide.SELL
                    )
                    adjusted_price = slippage_result.adjusted_price

                    # 计算交易金额和成本
                    trade_amount = adjusted_price * position.quantity
                    cost_breakdown = self.cost_calculator.calculate_sell_cost(trade_amount)
                    total_cost = cost_breakdown.total

                    # 计算盈亏（扣除成本）
                    gross_pnl = (adjusted_price - Decimal(str(position.entry_price))) * position.quantity
                    net_pnl = gross_pnl - total_cost
                    gross_pnl_float = float(gross_pnl)

                    # 盈亏比例（基于原始投入）
                    entry_amount = Decimal(str(position.entry_price)) * position.quantity
                    pnl_ratio = float(net_pnl / entry_amount) if entry_amount > 0 else 0

                    # 实际回收资金
                    actual_proceeds = trade_amount - total_cost
                    capital += actual_proceeds
                    total_costs += total_cost
                    total_slippage += slippage_result.slippage_amount * position.quantity

                    trade = Trade(
                        entry_date=position.entry_date,
                        exit_date=signal.date,
                        entry_price=position.entry_price,
                        exit_price=float(adjusted_price),
                        raw_entry_price=getattr(position, "_raw_entry_price", position.entry_price),
                        raw_exit_price=float(raw_price),
                        quantity=position.quantity,
                        trade_type=position.position_type,
                        pnl=float(net_pnl),
                        pnl_ratio=pnl_ratio,
                        gross_pnl=gross_pnl_float,
                        total_cost=float(total_cost),
                        buy_cost=getattr(position, "_buy_cost", None),
                        sell_cost={
                            "commission": float(cost_breakdown.commission),
                            "stamp_duty": float(cost_breakdown.stamp_duty),
                            "transfer_fee": float(cost_breakdown.transfer_fee),
                            "total": float(total_cost),
                        },
                        buy_slippage=getattr(position, "_buy_slippage", 0.0),
                        sell_slippage=float(slippage_result.slippage_rate),
                        reason_exit=signal.reason or "",
                    )
                    trades.append(trade)
                    position = None
                    position_buy_date = None  # 重置买入日期

                    # 更新最大回撤
                    if capital > max_capital:
                        max_capital = capital
                    drawdown = float((max_capital - capital) / max_capital)
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

        # 处理最后未平仓的情况
        if position is not None:
            last_price = Decimal(str(df.iloc[-1]["close"]))
            last_date = str(df.iloc[-1]["trade_date"])

            # 应用滑点
            slippage_result = self.slippage_model.apply_slippage(
                last_price, TradeSide.SELL
            )
            adjusted_price = slippage_result.adjusted_price

            # 计算交易金额和成本
            trade_amount = adjusted_price * position.quantity
            cost_breakdown = self.cost_calculator.calculate_sell_cost(trade_amount)
            total_cost = cost_breakdown.total

            gross_pnl = (adjusted_price - Decimal(str(position.entry_price))) * position.quantity
            net_pnl = gross_pnl - total_cost
            gross_pnl_float = float(gross_pnl)

            entry_amount = Decimal(str(position.entry_price)) * position.quantity
            pnl_ratio = float(net_pnl / entry_amount) if entry_amount > 0 else 0

            actual_proceeds = trade_amount - total_cost
            capital += actual_proceeds
            total_costs += total_cost
            total_slippage += slippage_result.slippage_amount * position.quantity

            trade = Trade(
                entry_date=position.entry_date,
                exit_date=last_date,
                entry_price=position.entry_price,
                exit_price=float(adjusted_price),
                raw_entry_price=getattr(position, "_raw_entry_price", position.entry_price),
                raw_exit_price=float(last_price),
                quantity=position.quantity,
                trade_type=position.position_type,
                pnl=float(net_pnl),
                pnl_ratio=pnl_ratio,
                gross_pnl=gross_pnl_float,
                total_cost=float(total_cost),
                buy_cost=getattr(position, "_buy_cost", None),
                sell_cost={
                    "commission": float(cost_breakdown.commission),
                    "stamp_duty": float(cost_breakdown.stamp_duty),
                    "transfer_fee": float(cost_breakdown.transfer_fee),
                    "total": float(total_cost),
                },
                buy_slippage=getattr(position, "_buy_slippage", 0.0),
                sell_slippage=float(slippage_result.slippage_rate),
                reason_exit="期末平仓",
            )
            trades.append(trade)

            if capital > max_capital:
                max_capital = capital
            drawdown = float((max_capital - capital) / max_capital)
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 计算统计指标
        final_capital = float(capital)
        total_return = (final_capital - self.initial_capital) / self.initial_capital

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
            start_date=str(df.iloc[0]["trade_date"]),
            end_date=str(df.iloc[-1]["trade_date"]),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            trades=trades,
            trade_count=len(trades),
            win_count=win_count,
            lose_count=lose_count,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_costs=float(total_costs),
            total_slippage=float(total_slippage),
            cost_config=self.cost_config,
            t1_violations=t1_violations,
            t1_skipped_sells=t1_skipped_sells,
        )
