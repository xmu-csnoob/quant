"""
Risk service - 风控服务层
"""

from datetime import datetime
from typing import Optional
from loguru import logger

from src.api.schemas.risk import RiskConfig, RiskStatus, RiskAction


# 内存存储（实际应用中应使用数据库）
_risk_config_store: Optional[RiskConfig] = None


def _get_default_config() -> RiskConfig:
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


def get_risk_config() -> RiskConfig:
    """获取当前风控配置"""
    global _risk_config_store
    if _risk_config_store is None:
        _risk_config_store = _get_default_config()
    return _risk_config_store


def save_risk_config(config: RiskConfig) -> RiskConfig:
    """保存风控配置"""
    global _risk_config_store
    _risk_config_store = config
    logger.info(f"风控配置已更新: {config.model_dump_json()}")
    return _risk_config_store


def get_risk_status() -> RiskStatus:
    """获取风控实时状态"""
    config = get_risk_config()

    # 模拟交易统计数据（实际应从交易系统获取）
    global _trading_stats

    if not hasattr(_trading_stats, '_daily_trades'):
        _trading_stats._daily_trades = [
            {"code": "600519.SH", "shares": 100, "price": 1680.00},
            {"code": "000858.SZ", "shares": 300, "price": 145.80},
        ]
        _trading_stats._total_equity = 1000000
        _trading_stats._daily_loss = 0

    # 计算风险等级
    risk_level = "low"
    if config.max_drawdown > 10:
        risk_level = "medium"
    if config.max_drawdown > 20:
        risk_level = "high"

    # 计算T+1锁定数量
    t1_locked = sum(pos.get('shares', 0) - pos.get('available', 0) for pos in _trading_stats._daily_trades)

    return RiskStatus(
        current_positions=len(_trading_stats._daily_trades),
        position_ratio=30.0,  # 简化：假设2个持仓
        daily_loss=abs(_trading_stats._daily_loss),
        daily_loss_ratio=abs(_trading_stats._daily_loss) / _trading_stats._total_equity * 100,
        max_drawdown=config.max_drawdown,
        consecutive_losses=0,
        t1_locked_shares=t1_locked,
        risk_level=risk_level,
        timestamp=datetime.now().isoformat()
    )


# 模拟交易统计数据（实际应从交易系统获取）
_trading_stats = type('SimpleNamespace', object)


if __name__ == "__main__":
    print("Risk service module loaded")
    logger.info("Risk service initialized")
