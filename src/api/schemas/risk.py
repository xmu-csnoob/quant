"""
Risk schemas - 风控数据模型
"""

from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    """风控配置"""
    # 仓位限制
    max_position_count: int = Field(default=3, ge=1, le=10, description="最大持仓数量")
    max_position_ratio: float = Field(default=30.0, ge=5, le=50, description="单股最大仓位(%)")
    min_position_ratio: float = Field(default=5.0, ge=1, le=20, description="单股最小仓位(%)")

    # 止损止盈
    stop_loss_ratio: float = Field(default=10.0, ge=1, le=50, description="止损比例(%)")
    take_profit_ratio: float = Field(default=20.0, ge=5, le=100, description="止盈比例(%)")
    enable_auto_stop_loss: bool = Field(default=True, description="启用自动止损")
    enable_trailing_stop: bool = Field(default=False, description="启用移动止损")
    trailing_stop_ratio: float = Field(default=5.0, ge=1, le=20, description="移动止损回撤(%)")

    # 风控阈值
    max_daily_loss: float = Field(default=5.0, ge=1, le=20, description="单日最大亏损(%)")
    max_drawdown: float = Field(default=15.0, ge=5, le=50, description="最大回撤(%)")
    enable_consecutive_loss: bool = Field(default=True, description="启用连续亏损保护")
    max_consecutive_losses: int = Field(default=3, ge=1, le=10, description="最大连续亏损次数")

    # T+1规则
    enable_t1_rule: bool = Field(default=True, description="启用T+1规则")


class RiskStatus(BaseModel):
    """风控实时状态"""
    current_positions: int = Field(description="当前持仓数量")
    position_ratio: float = Field(description="当前仓位比例(%)")
    daily_loss: float = Field(default=0.0, description="今日亏损金额")
    daily_loss_ratio: float = Field(default=0.0, description="今日亏损率(%)")
    max_drawdown: float = Field(default=0.0, description="最大回撤(%)")
    consecutive_losses: int = Field(default=0, description="连续亏损次数")
    t1_locked_shares: int = Field(default=0, description="T+1锁定数量")
    risk_level: str = Field(default="low", description="风险等级: low/medium/high")
    timestamp: str = Field(description="更新时间")


class RiskAction(BaseModel):
    """风控动作"""
    action: str = Field(description="动作类型")
    detail: str = Field(description="详细信息")
    timestamp: str = Field(description="执行时间")
