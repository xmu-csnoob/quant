"""
风险管理器

实时监控和管理交易风险
"""

import pandas as pd
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from src.risk.position_sizer import PositionSizer, PositionSize


class RiskAction(Enum):
    """风险动作"""
    HOLD = "持有"
    REDUCE = "减仓"
    CLOSE = "平仓"
    STOP_TRADING = "停止交易"


@dataclass
class RiskCheck:
    """风险检查结果"""
    passed: bool
    action: RiskAction
    reason: str
    current_position_ratio: float
    current_drawdown: float
    total_pnl: float


@dataclass
class Position:
    """持仓信息"""
    entry_price: float
    shares: int
    entry_date: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0
    unrealized_pnl: float = 0
    unrealized_pnl_ratio: float = 0


class RiskManager:
    """
    风险管理器

    功能：
    1. 止损止盈检查
    2. 最大回撤控制
    3. 单笔/总仓位限制
    4. 连续亏损保护
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_sizer: PositionSizer = None,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
        max_position_ratio: float = 0.95,
        max_total_position_ratio: float = 1.0,
        max_drawdown: float = 0.20,
        max_daily_loss: float = 0.05,
        max_consecutive_losses: int = 3,
    ):
        """
        初始化

        Args:
            initial_capital: 初始资金
            position_sizer: 仓位管理器
            stop_loss: 单笔止损比例
            take_profit: 单笔止盈比例
            max_position_ratio: 单个持仓最大比例
            max_total_position_ratio: 总仓位最大比例
            max_drawdown: 最大回撤限制
            max_daily_loss: 单日最大亏损
            max_consecutive_losses: 最大连续亏损次数
        """
        self.initial_capital = initial_capital
        self.position_sizer = position_sizer or PositionSizer(initial_capital)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_ratio = max_position_ratio
        self.max_total_position_ratio = max_total_position_ratio
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses

        # 状态跟踪
        self.positions: dict[str, Position] = {}
        self.closed_trades: List[dict] = []
        self.consecutive_losses = 0
        self.peak_capital = initial_capital
        self.daily_pnl = 0
        self.current_capital = initial_capital

    def check_entry(
        self,
        price: float,
        signal_confidence: float,
        stock_code: str,
    ) -> tuple[bool, PositionSize, str]:
        """
        检查是否允许开仓

        Returns:
            (是否允许, 仓位信息, 原因)
        """
        # 1. 检查是否达到最大回撤限制
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.max_drawdown:
            return False, PositionSize(0, 0, 0, ""), f"达到最大回撤限制（{current_drawdown:.1%}）"

        # 2. 检查连续亏损次数
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, PositionSize(0, 0, 0, ""), f"达到最大连续亏损次数（{self.consecutive_losses}次）"

        # 3. 检查单日亏损
        daily_loss_ratio = abs(self.daily_pnl) / self.initial_capital if self.daily_pnl < 0 else 0
        if daily_loss_ratio > self.max_daily_loss:
            return False, PositionSize(0, 0, 0, ""), f"达到单日最大亏损（{daily_loss_ratio:.1%}）"

        # 4. 检查总仓位
        total_position_value = sum(p.shares * p.current_price for p in self.positions.values() if p.shares > 0)
        total_position_ratio = total_position_value / self.initial_capital

        if total_position_ratio >= self.max_total_position_ratio:
            return False, PositionSize(0, 0, 0, ""), f"达到最大总仓位（{total_position_ratio:.1%}）"

        # 5. 计算建议仓位
        position_size = self.position_sizer.calculate(
            price=price,
            capital=self.current_capital,
            confidence=signal_confidence,
        )

        # 6. 检查单笔仓位限制
        if position_size.risk_ratio > self.max_position_ratio:
            position_size.risk_ratio = self.max_position_ratio
            position_size.amount = self.current_capital * self.max_position_ratio
            position_size.shares = int(position_size.amount / price / 100) * 100

        return True, position_size, "允许开仓"

    def check_exit(
        self,
        stock_code: str,
        current_price: float,
        date: str,
    ) -> RiskCheck:
        """
        检查是否需要平仓

        Returns:
            风险检查结果
        """
        if stock_code not in self.positions:
            return RiskCheck(
                passed=True,
                action=RiskAction.HOLD,
                reason="无持仓",
                current_position_ratio=0,
                current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
                total_pnl=sum(p.unrealized_pnl for p in self.positions.values()),
            )

        position = self.positions[stock_code]
        position.current_price = current_price

        # 计算未实现盈亏
        position.unrealized_pnl = (current_price - position.entry_price) * position.shares
        position.unrealized_pnl_ratio = (current_price - position.entry_price) / position.entry_price

        # 1. 检查止损
        if position.stop_loss:
            if current_price <= position.stop_loss:
                return RiskCheck(
                    passed=False,
                    action=RiskAction.CLOSE,
                    reason=f"触发止损（价格{current_price:.2f} <= 止损{position.stop_loss:.2f}）",
                    current_position_ratio=position.shares * current_price / self.initial_capital,
                    current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
                    total_pnl=position.unrealized_pnl,
                )

        # 2. 检查比例止损
        if position.unrealized_pnl_ratio <= -self.stop_loss:
            return RiskCheck(
                passed=False,
                action=RiskAction.CLOSE,
                reason=f"触发比例止损（亏损{position.unrealized_pnl_ratio:.1%} <= {self.stop_loss:.1%}）",
                current_position_ratio=position.shares * current_price / self.initial_capital,
                current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
                total_pnl=position.unrealized_pnl,
            )

        # 3. 检查止盈
        if position.take_profit:
            if current_price >= position.take_profit:
                return RiskCheck(
                    passed=False,
                    action=RiskAction.CLOSE,
                    reason=f"触发止盈（价格{current_price:.2f} >= 目标{position.take_profit:.2f}）",
                    current_position_ratio=position.shares * current_price / self.initial_capital,
                    current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
                    total_pnl=position.unrealized_pnl,
                )

        # 4. 检查比例止盈
        if position.unrealized_pnl_ratio >= self.take_profit:
            return RiskCheck(
                passed=False,
                action=RiskAction.CLOSE,
                reason=f"触发比例止盈（盈利{position.unrealized_pnl_ratio:.1%} >= {self.take_profit:.1%}）",
                current_position_ratio=position.shares * current_price / self.initial_capital,
                current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
                total_pnl=position.unrealized_pnl,
            )

        # 5. 检查移动止损（回撤50%平仓）
        if position.unrealized_pnl_ratio > 0.05:
            # 盈利超过5%，设置移动止损
            trailing_stop = position.unrealized_pnl_ratio * 0.5
            if position.unrealized_pnl_ratio < trailing_stop:
                return RiskCheck(
                    passed=False,
                    action=RiskAction.CLOSE,
                    reason=f"触发移动止损（盈利回撤从{position.unrealized_pnl_ratio:.1%}到{trailing_stop:.1%}）",
                    current_position_ratio=position.shares * current_price / self.initial_capital,
                    current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
                    total_pnl=position.unrealized_pnl,
                )

        # 无需平仓
        return RiskCheck(
            passed=True,
            action=RiskAction.HOLD,
            reason="持有",
            current_position_ratio=position.shares * current_price / self.initial_capital,
            current_drawdown=(self.peak_capital - self.current_capital) / self.peak_capital,
            total_pnl=position.unrealized_pnl,
        )

    def open_position(
        self,
        stock_code: str,
        entry_price: float,
        shares: int,
        entry_date: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """开仓"""
        self.positions[stock_code] = Position(
            entry_price=entry_price,
            shares=shares,
            entry_date=entry_date,
            stop_loss=stop_loss or entry_price * (1 - self.stop_loss),
            take_profit=take_profit or entry_price * (1 + self.take_profit),
        )

        # 扣减资金
        self.current_capital -= shares * entry_price

        logger.info(
            f"开仓: {stock_code}, 价格={entry_price:.2f}, "
            f"数量={shares}, 止损={self.positions[stock_code].stop_loss:.2f}, "
            f"止盈={self.positions[stock_code].take_profit:.2f}"
        )

    def close_position(
        self,
        stock_code: str,
        exit_price: float,
        exit_date: str,
        reason: str,
    ):
        """平仓"""
        if stock_code not in self.positions:
            return

        position = self.positions[stock_code]

        # 计算盈亏
        pnl = (exit_price - position.entry_price) * position.shares
        pnl_ratio = (exit_price - position.entry_price) / position.entry_price

        # 更新资金
        self.current_capital += position.shares * exit_price

        # 更新峰值
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # 记录交易
        trade = {
            "stock_code": stock_code,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "pnl": pnl,
            "pnl_ratio": pnl_ratio,
            "entry_date": position.entry_date,
            "exit_date": exit_date,
            "reason": reason,
        }
        self.closed_trades.append(trade)

        # 更新连续亏损计数
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        logger.info(
            f"平仓: {stock_code}, 价格={exit_price:.2f}, "
            f"盈亏={pnl:.2f}({pnl_ratio:.1%}), 原因={reason}"
        )

        # 删除持仓
        del self.positions[stock_code]

        return trade

    def update_daily_pnl(self, pnl: float):
        """更新单日盈亏"""
        self.daily_pnl = pnl

    def get_summary(self) -> dict:
        """获取风险摘要"""
        total_position_value = sum(p.shares * p.current_price for p in self.positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        realized_pnl = sum(t["pnl"] for t in self.closed_trades)
        total_capital = self.current_capital + total_position_value

        return {
            "initial_capital": self.initial_capital,
            "current_capital": total_capital,
            "total_return": (total_capital / self.initial_capital - 1),
            "realized_pnl": realized_pnl,
            "unrealized_pnl": total_unrealized_pnl,
            "peak_capital": self.peak_capital,
            "current_drawdown": (self.peak_capital - total_capital) / self.peak_capital,
            "consecutive_losses": self.consecutive_losses,
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
        }
