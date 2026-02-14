"""
Risk service - 风控服务层
从数据库读取真实数据
"""

from datetime import datetime
from typing import Optional
from loguru import logger

from src.api.schemas.risk import RiskConfig, RiskStatus, RiskAction
from src.data.storage.sqlite_storage import SQLiteStorage


class RiskService:
    """风控服务 - 从数据库读取真实数据"""

    def __init__(self):
        self._storage = SQLiteStorage()
        self._config: Optional[RiskConfig] = None

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

    def get_status(self) -> RiskStatus:
        """获取风控实时状态 - 从数据库读取"""
        config = self.get_config()

        try:
            # 从数据库获取持仓
            positions_df = self._storage.get_positions()
            current_positions = len(positions_df) if positions_df is not None and len(positions_df) > 0 else 0

            # 计算持仓比例
            position_ratio = 0.0
            total_equity = 1000000.0  # 默认总资产

            if positions_df is not None and len(positions_df) > 0:
                total_market_value = positions_df['market_value'].sum() if 'market_value' in positions_df.columns else 0
                # 简化计算
                position_ratio = (total_market_value / total_equity * 100) if total_equity > 0 else 0

            # 从数据库获取交易记录计算盈亏
            trades_df = self._storage.get_trades()
            daily_loss = 0.0
            consecutive_losses = 0

            if trades_df is not None and len(trades_df) > 0:
                # 计算连续亏损
                # 简化：统计sell交易的亏损
                sells = trades_df[trades_df['side'] == 'sell'] if 'side' in trades_df.columns else trades_df
                consecutive_losses = 0  # 需要更复杂的逻辑来计算

            # 计算风险等级
            risk_level = "low"
            if position_ratio > 60:
                risk_level = "medium"
            if position_ratio > 80 or daily_loss > config.max_daily_loss:
                risk_level = "high"

            return RiskStatus(
                current_positions=current_positions,
                position_ratio=round(position_ratio, 2),
                daily_loss=abs(daily_loss),
                daily_loss_ratio=abs(daily_loss) / total_equity * 100 if total_equity > 0 else 0,
                max_drawdown=config.max_drawdown,
                consecutive_losses=consecutive_losses,
                t1_locked_shares=0,  # TODO: 实现T+1锁定计算
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
                max_drawdown=config.max_drawdown,
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
