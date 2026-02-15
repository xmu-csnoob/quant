"""
Risk service - 风控服务层
从数据库读取真实数据
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger

from src.api.schemas.risk import RiskConfig, RiskStatus, RiskAction
from src.data.storage.sqlite_storage import SQLiteStorage


class RiskService:
    """风控服务 - 从数据库读取真实数据"""

    def __init__(self):
        self._storage = SQLiteStorage()
        self._config: Optional[RiskConfig] = None
        self._peak_equity: float = 0  # 峰值权益，用于计算回撤

    def _get_default_config(self) -> RiskConfig:
        """获取默认配置"""
        return RiskConfig(
            max_position_count=3,
            max_position_ratio=30.0,
            min_position_ratio=5.0,
            stop_loss_ratio=10.0,
            take_profit_ratio=20.0,
            enable_auto_stop_loss=True,
            enable_trailing_stop=False,
            trailing_stop_ratio=5.0,
            max_daily_loss=5.0,
            max_drawdown=15.0,
            enable_consecutive_loss=True,
            max_consecutive_losses=3,
            enable_t1_rule=True,
        )

    def get_config(self) -> RiskConfig:
        """获取当前风控配置"""
        if self._config is None:
            self._config = self._get_default_config()
        return self._config

    def save_config(self, config: RiskConfig) -> RiskConfig:
        """保存风控配置"""
        self._config = config
        logger.info(f"风控配置已更新: {config.model_dump_json()}")
        return self._config

    def _get_total_equity(self) -> float:
        """从数据库获取总权益（现金 + 持仓市值）"""
        try:
            # 获取账户资金
            balance = self._storage.get_account_balance()
            cash = balance.get("cash", 0)
            initial_capital = balance.get("initial_capital", 1000000)

            # 获取持仓市值
            positions_df = self._storage.get_positions()
            market_value = 0
            if positions_df is not None and len(positions_df) > 0:
                market_value = positions_df['market_value'].sum() if 'market_value' in positions_df.columns else 0

            total_equity = cash + market_value

            # 更新峰值权益
            if total_equity > self._peak_equity:
                self._peak_equity = total_equity

            return total_equity if total_equity > 0 else initial_capital
        except Exception as e:
            logger.error(f"获取总权益失败: {e}")
            return 1000000.0

    def _calculate_daily_loss(self) -> float:
        """
        计算当日亏损

        基于持仓表的unrealized_pnl变化来估算，或基于交易记录计算
        注意：trades表的trade_date格式为 'YYYY-MM-DD HH:MM:SS'
        """
        try:
            trades_df = self._storage.get_trades()
            if trades_df is None or len(trades_df) == 0:
                return 0.0

            # 获取今天的日期（两种格式都要匹配）
            today_ymd = datetime.now().strftime("%Y-%m-%d")  # "2024-01-15"
            today_ymd2 = datetime.now().strftime("%Y%m%d")     # "20240115"

            # 筛选今天的卖出交易（兼容两种日期格式）
            if 'trade_date' in trades_df.columns and 'side' in trades_df.columns:
                # 匹配 "2024-01-15" 或以 "20240115" 开头
                mask = (
                    trades_df['trade_date'].astype(str).str.startswith(today_ymd) |
                    trades_df['trade_date'].astype(str).str.startswith(today_ymd2)
                )
                today_trades = trades_df[mask]
                sell_trades = today_trades[today_trades['side'] == 'sell']

                # 计算已实现亏损（基于买卖价差估算）
                # 由于trades表没有realized_pnl，需要从positions表计算
                # 简化方案：使用持仓表的unrealized_pnl变化
                daily_loss = 0.0

                # 尝试从positions表获取当日PnL变化
                positions_df = self._storage.get_positions()
                if positions_df is not None and len(positions_df) > 0:
                    if 'unrealized_pnl' in positions_df.columns:
                        # 负的unrealized_pnl代表浮亏
                        daily_loss = abs(positions_df[positions_df['unrealized_pnl'] < 0]['unrealized_pnl'].sum())

                return daily_loss

            return 0.0
        except Exception as e:
            logger.error(f"计算当日亏损失败: {e}")
            return 0.0

    def _calculate_consecutive_losses(self) -> int:
        """
        计算连续亏损次数

        基于持仓表和交易记录分析
        注意：trades表没有pnl列，需要通过其他方式估算
        """
        try:
            # 从持仓表检查是否有浮亏持仓
            positions_df = self._storage.get_positions()
            if positions_df is None or len(positions_df) == 0:
                return 0

            consecutive_losses = 0

            # 检查当前持仓是否有亏损
            if 'unrealized_pnl' in positions_df.columns:
                for pnl in positions_df['unrealized_pnl']:
                    if pnl < 0:
                        consecutive_losses += 1

            # 简化：返回亏损持仓数量作为"连续亏损"的近似
            # TODO: 实现更精确的连续亏损天数计算，需要在trades表添加realized_pnl列
            return consecutive_losses
        except Exception as e:
            logger.error(f"计算连续亏损次数失败: {e}")
            return 0

    def _calculate_t1_locked_shares(self) -> int:
        """计算T+1锁定的股数（当日买入不可卖）"""
        try:
            positions_df = self._storage.get_positions()
            if positions_df is None or len(positions_df) == 0:
                return 0

            # 获取今天的日期
            today = datetime.now().strftime("%Y%m%d")

            # 检查持仓中今日买入的部分
            # 注意：简化实现，实际需要在持仓记录中追踪买入日期
            # 这里返回0，因为当前数据库schema没有追踪买入日期
            # TODO: 在positions表中添加buy_date字段以支持T+1追踪
            return 0
        except Exception as e:
            logger.error(f"计算T+1锁定股数失败: {e}")
            return 0

    def _calculate_max_drawdown(self) -> float:
        """计算当前最大回撤"""
        try:
            total_equity = self._get_total_equity()
            if self._peak_equity > 0:
                drawdown = (self._peak_equity - total_equity) / self._peak_equity * 100
                return max(0, drawdown)
            return 0.0
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0

    def get_status(self) -> RiskStatus:
        """获取风控实时状态 - 从数据库读取"""
        config = self.get_config()

        try:
            # 从数据库获取持仓
            positions_df = self._storage.get_positions()
            current_positions = len(positions_df) if positions_df is not None and len(positions_df) > 0 else 0

            # 获取总权益
            total_equity = self._get_total_equity()

            # 计算持仓市值和比例
            position_ratio = 0.0
            if positions_df is not None and len(positions_df) > 0:
                total_market_value = positions_df['market_value'].sum() if 'market_value' in positions_df.columns else 0
                position_ratio = (total_market_value / total_equity * 100) if total_equity > 0 else 0

            # 计算当日亏损
            daily_loss = self._calculate_daily_loss()
            daily_loss_ratio = (daily_loss / total_equity * 100) if total_equity > 0 else 0

            # 计算连续亏损次数
            consecutive_losses = self._calculate_consecutive_losses()

            # 计算T+1锁定股数
            t1_locked_shares = self._calculate_t1_locked_shares()

            # 计算当前回撤
            current_drawdown = self._calculate_max_drawdown()

            # 计算风险等级
            risk_level = "low"
            if position_ratio > 60 or consecutive_losses >= 2:
                risk_level = "medium"
            if position_ratio > 80 or daily_loss_ratio > config.max_daily_loss or consecutive_losses >= config.max_consecutive_losses or current_drawdown > config.max_drawdown:
                risk_level = "high"

            return RiskStatus(
                current_positions=current_positions,
                position_ratio=round(position_ratio, 2),
                daily_loss=round(daily_loss, 2),
                daily_loss_ratio=round(daily_loss_ratio, 2),
                max_drawdown=round(current_drawdown, 2),  # 返回当前回撤，不是阈值
                consecutive_losses=consecutive_losses,
                t1_locked_shares=t1_locked_shares,
                risk_level=risk_level,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"获取风控状态失败: {e}")
            # 返回默认状态
            return RiskStatus(
                current_positions=0,
                position_ratio=0.0,
                daily_loss=0.0,
                daily_loss_ratio=0.0,
                max_drawdown=0.0,  # 当前回撤为0，不是阈值
                consecutive_losses=0,
                t1_locked_shares=0,
                risk_level="low",
                timestamp=datetime.now().isoformat()
            )


# 单例
risk_service = RiskService()


# 兼容旧接口
def get_risk_config() -> RiskConfig:
    return risk_service.get_config()


def save_risk_config(config: RiskConfig) -> RiskConfig:
    return risk_service.save_config(config)


def get_risk_status() -> RiskStatus:
    return risk_service.get_status()
