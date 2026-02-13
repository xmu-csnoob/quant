"""
Account service - 账户服务层
提供账户相关的业务逻辑

包含T+1规则：
- 从T+1管理器获取可卖数量
- 正确显示锁定持仓
"""

from datetime import date
from typing import List
import random

from src.api.schemas.account import AccountSummary, Position
from src.trading.t1_manager import get_t1_manager


class AccountService:
    """账户服务（含T+1规则支持）"""

    def __init__(self):
        self._t1_manager = get_t1_manager()

        # 模拟数据 - 股票名称映射
        self._stock_names = {
            "600000.SH": "浦发银行",
            "600519.SH": "贵州茅台",
            "000858.SZ": "五粮液",
            "601318.SH": "中国平安",
        }

        # 模拟数据 - 当前价格（实际应从行情接口获取）
        self._current_prices = {
            "600000.SH": 10.58,
            "600519.SH": 1725.50,
            "000858.SZ": 148.20,
            "601318.SH": 41.85,
        }

        # 账户概览数据
        self._cash = 456789.30

    def get_summary(self) -> AccountSummary:
        """获取账户概览"""
        # 从T+1管理器获取所有持仓
        positions = self.get_positions()

        # 计算市值
        market_value = sum(pos.market_value for pos in positions)
        total_assets = self._cash + market_value

        # 计算总盈亏（简化：基于持仓盈亏）
        total_profit = sum(pos.profit for pos in positions)

        # 模拟今日收益波动
        fluctuation = random.uniform(-0.1, 0.1)
        today_return = round(0.42 + fluctuation, 2)
        today_profit = round(total_assets * today_return / 100, 2)

        return AccountSummary(
            total_assets=round(total_assets, 2),
            cash=self._cash,
            market_value=round(market_value, 2),
            total_profit=round(total_profit, 2),
            total_return=round(total_profit / (total_assets - total_profit) * 100, 2) if total_assets > total_profit else 0,
            today_profit=today_profit,
            today_return=today_return
        )

    def get_positions(self) -> List[Position]:
        """获取持仓列表（包含正确的T+1可卖数量）"""
        positions = []
        today = date.today()

        # 从T+1管理器获取所有持仓
        all_positions = self._t1_manager.get_all_positions(today)

        for code, pos_info in all_positions.items():
            total_shares = pos_info["total_shares"]
            if total_shares <= 0:
                continue

            # 获取当前价格（模拟波动）
            base_price = self._current_prices.get(code, 100.0)
            price_change = random.uniform(-0.5, 0.5)
            current_price = round(base_price * (1 + price_change / 100), 2)

            # 获取可卖数量（T+1）
            available = pos_info["available_shares"]
            avg_cost = pos_info["average_cost"]

            # 计算市值和盈亏
            market_value = round(current_price * total_shares, 2)
            profit = round((current_price - avg_cost) * total_shares, 2)
            profit_ratio = round((current_price - avg_cost) / avg_cost * 100, 2) if avg_cost > 0 else 0

            positions.append(Position(
                code=code,
                name=self._stock_names.get(code, "未知股票"),
                shares=total_shares,
                available=available,  # T+1可卖数量
                cost_price=round(avg_cost, 2),
                current_price=current_price,
                market_value=market_value,
                profit=profit,
                profit_ratio=profit_ratio,
                weight=0  # 稍后计算
            ))

        # 计算权重
        total_value = sum(p.market_value for p in positions)
        for pos in positions:
            pos.weight = round(pos.market_value / total_value * 100, 2) if total_value > 0 else 0

        return positions


# 单例
account_service = AccountService()
