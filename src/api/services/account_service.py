"""
Account service - 账户服务层
支持虚拟账户(paper)和实盘账户(live)
"""

from datetime import date, datetime
from typing import List, Optional
from loguru import logger

from src.api.schemas.account import AccountSummary, Position
from src.data.storage.sqlite_storage import SQLiteStorage


class AccountService:
    """账户服务 - 支持虚拟/实盘账户"""

    def __init__(self, account_type: str = "paper"):
        """
        初始化账户服务

        Args:
            account_type: 账户类型 'paper'(虚拟) 或 'live'(实盘)
        """
        self._storage = SQLiteStorage()
        self._account_type = account_type

    def get_stock_name(self, code: str) -> str:
        """从数据库获取股票名称"""
        try:
            df = self._storage.get_daily_prices(code)
            if df is not None and len(df) > 0 and 'name' in df.columns:
                return df.iloc[0]['name']
        except Exception as e:
            logger.warning(f"获取股票名称失败 {code}: {e}")
        return code

    def get_latest_price(self, code: str) -> Optional[float]:
        """从数据库获取最新价格"""
        try:
            df = self._storage.get_daily_prices(code)
            if df is not None and len(df) > 0:
                df = df.sort_values('trade_date', ascending=False)
                return float(df.iloc[0]['close'])
        except Exception as e:
            logger.warning(f"获取最新价格失败 {code}: {e}")
        return None

    def get_summary(self) -> AccountSummary:
        """获取账户概览"""
        positions = self.get_positions()

        # 计算市值
        market_value = sum(pos.market_value for pos in positions)

        # 从数据库获取现金
        balance = self._storage.get_account_balance(self._account_type)
        cash = balance.get("cash", 1000000.0)
        initial_capital = balance.get("initial_capital", 1000000.0)

        total_assets = cash + market_value

        # 计算总盈亏
        total_profit = total_assets - initial_capital
        total_return = (total_profit / initial_capital * 100) if initial_capital > 0 else 0

        # 持仓盈亏
        position_profit = sum(pos.profit for pos in positions)

        # 今日盈亏（简化）
        today_profit = 0.0
        today_return = 0.0

        return AccountSummary(
            total_assets=round(total_assets, 2),
            cash=round(cash, 2),
            market_value=round(market_value, 2),
            total_profit=round(total_profit, 2),
            total_return=round(total_return, 2),
            today_profit=round(today_profit, 2),
            today_return=round(today_return, 2)
        )

    def get_positions(self) -> List[Position]:
        """获取持仓列表"""
        positions = []

        try:
            df = self._storage.get_positions(self._account_type)
            if df is None or len(df) == 0:
                return positions

            for _, row in df.iterrows():
                code = row.get('symbol', '')
                total_shares = int(row.get('quantity', 0))

                if total_shares <= 0:
                    continue

                current_price = self.get_latest_price(code)
                if current_price is None:
                    current_price = float(row.get('current_price', 0))

                avg_cost = float(row.get('avg_cost', 0))

                market_value = round(current_price * total_shares, 2)
                profit = round((current_price - avg_cost) * total_shares, 2)
                profit_ratio = round((current_price - avg_cost) / avg_cost * 100, 2) if avg_cost > 0 else 0

                positions.append(Position(
                    code=code,
                    name=self.get_stock_name(code),
                    shares=total_shares,
                    available=total_shares,  # TODO: T+1
                    cost_price=round(avg_cost, 2),
                    current_price=round(current_price, 2),
                    market_value=market_value,
                    profit=profit,
                    profit_ratio=profit_ratio,
                    weight=0
                ))

            # 计算权重
            total_value = sum(p.market_value for p in positions)
            for pos in positions:
                pos.weight = round(pos.market_value / total_value * 100, 2) if total_value > 0 else 0

        except Exception as e:
            logger.error(f"获取持仓失败: {e}")

        return positions

    def update_cash(self, delta: float):
        """更新现金余额（买入减少，卖出增加）"""
        balance = self._storage.get_account_balance(self._account_type)
        new_cash = balance.get("cash", 1000000.0) + delta
        self._storage.update_account_balance(new_cash, self._account_type)


# 单例 - 默认虚拟账户
paper_account_service = AccountService("paper")
# 实盘账户（暂未实现券商API）
live_account_service = AccountService("live")

# 默认使用虚拟账户
account_service = paper_account_service
