"""
仓位管理器

根据风险水平动态调整仓位大小
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class PositionSize:
    """仓位大小信息"""
    shares: int        # 股数（手）
    amount: float      # 金额
    risk_ratio: float  # 风险比例（0-1）
    reason: str        # 原因说明


class PositionSizer:
    """
    仓位管理器

    方法：
    1. 固定金额法
    2. 固定比例法
    3. 风险平价法（凯利公式）
    4. ATR止损法
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        method: str = "fixed_ratio",
        max_position_ratio: float = 0.95,
        min_position_ratio: float = 0.05,
    ):
        """
        初始化

        Args:
            initial_capital: 初始资金
            method: 仓位计算方法
                - fixed_amount: 固定金额
                - fixed_ratio: 固定比例
                - kelly: 凯利公式
                - atr: ATR止损法
            max_position_ratio: 最大仓位比例
            min_position_ratio: 最小仓位比例
        """
        self.initial_capital = initial_capital
        self.method = method
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio

    def calculate(
        self,
        price: float,
        capital: float,
        stop_loss: Optional[float] = None,
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss: Optional[float] = None,
        confidence: float = 0.5,
    ) -> PositionSize:
        """
        计算仓位大小

        Args:
            price: 当前价格
            capital: 可用资金
            stop_loss: 止损价格
            atr: ATR值
            win_rate: 历史胜率
            avg_win_loss: 平均盈亏比
            confidence: 信号置信度（0-1）

        Returns:
            仓位信息
        """
        if self.method == "fixed_amount":
            return self._fixed_amount(price, capital)
        elif self.method == "fixed_ratio":
            return self._fixed_ratio(price, capital, confidence)
        elif self.method == "kelly":
            return self._kelly(price, capital, win_rate, avg_win_loss, confidence)
        elif self.method == "atr":
            return self._atr_risk(price, capital, stop_loss, atr, confidence)
        else:
            logger.warning(f"Unknown method: {self.method}, using fixed_ratio")
            return self._fixed_ratio(price, capital, confidence)

    def _fixed_amount(self, price: float, capital: float) -> PositionSize:
        """固定金额法"""
        # 每次交易固定金额（如初始资金的10%）
        trade_amount = self.initial_capital * 0.1

        # 限制不超过可用资金的95%
        trade_amount = min(trade_amount, capital * 0.95)

        shares = int(trade_amount / price / 100) * 100  # 整手
        actual_amount = shares * price
        risk_ratio = actual_amount / self.initial_capital

        return PositionSize(
            shares=shares,
            amount=actual_amount,
            risk_ratio=risk_ratio,
            reason="固定金额法",
        )

    def _fixed_ratio(
        self,
        price: float,
        capital: float,
        confidence: float,
    ) -> PositionSize:
        """
        固定比例法

        根据信号置信度调整仓位比例：
        - 置信度 > 0.8: 90%仓位
        - 置信度 > 0.6: 70%仓位
        - 置信度 > 0.4: 50%仓位
        - 其他: 30%仓位
        """
        if confidence > 0.8:
            ratio = 0.9
        elif confidence > 0.6:
            ratio = 0.7
        elif confidence > 0.4:
            ratio = 0.5
        else:
            ratio = 0.3

        # 限制在[min, max]范围内
        ratio = max(self.min_position_ratio, min(self.max_position_ratio, ratio))

        trade_amount = capital * ratio
        shares = int(trade_amount / price / 100) * 100
        actual_amount = shares * price

        return PositionSize(
            shares=shares,
            amount=actual_amount,
            risk_ratio=ratio,
            reason=f"固定比例法（置信度{confidence:.1%}）",
        )

    def _kelly(
        self,
        price: float,
        capital: float,
        win_rate: Optional[float],
        avg_win_loss: Optional[float],
        confidence: float,
    ) -> PositionSize:
        """
        凯利公式

        f* = (bp - q) / b

        其中：
        - f* = 最优仓位比例
        - b = 平均盈利/平均亏损
        - p = 胜率
        - q = 败率 = 1-p
        """
        # 默认值
        if win_rate is None:
            win_rate = 0.5
        if avg_win_loss is None:
            avg_win_loss = 1.5

        # 凯利公式
        kelly_ratio = (avg_win_loss * win_rate - (1 - win_rate)) / avg_win_loss

        # 凯利公式通常建议半凯利（更保守）
        kelly_ratio *= 0.5

        # 结合信号置信度
        kelly_ratio *= confidence

        # 限制范围
        ratio = max(self.min_position_ratio, min(self.max_position_ratio, kelly_ratio))

        trade_amount = capital * ratio
        shares = int(trade_amount / price / 100) * 100
        actual_amount = shares * price

        return PositionSize(
            shares=shares,
            amount=actual_amount,
            risk_ratio=ratio,
            reason=f"凯利公式（胜率{win_rate:.1%}，盈亏比{avg_win_loss:.2f}）",
        )

    def _atr_risk(
        self,
        price: float,
        capital: float,
        stop_loss: Optional[float],
        atr: Optional[float],
        confidence: float,
    ) -> PositionSize:
        """
        ATR止损法

        根据ATR计算止损距离，再根据风险预算计算仓位

        仓位 = (资金 × 风险比例) / (价格 × ATR倍数)
        """
        if atr is None:
            # 如果没有ATR，回退到固定比例
            return self._fixed_ratio(price, capital, confidence)

        # 假设愿意承受单笔2%的风险
        risk_per_trade = 0.02
        atr_multiplier = 2  # 2倍ATR作为止损距离

        # 止损距离
        stop_distance = atr * atr_multiplier

        # 计算仓位
        risk_amount = capital * risk_per_trade * confidence
        shares = int(risk_amount / stop_distance / 100) * 100

        # 限制不超过可用资金
        max_shares = int(capital * 0.95 / price / 100) * 100
        shares = min(shares, max_shares)

        actual_amount = shares * price
        risk_ratio = actual_amount / self.initial_capital

        return PositionSize(
            shares=shares,
            amount=actual_amount,
            risk_ratio=risk_ratio,
            reason=f"ATR止损法（ATR={atr:.2f}，止损距离={stop_distance:.2f}）",
        )
