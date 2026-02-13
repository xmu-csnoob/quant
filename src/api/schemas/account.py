"""
Account related schemas.
账户相关数据模型
"""

from typing import Optional
from pydantic import BaseModel, Field


class AccountSummary(BaseModel):
    """账户概览"""
    total_assets: float = Field(description="总资产")
    cash: float = Field(description="现金")
    market_value: float = Field(description="持仓市值")
    total_profit: float = Field(description="总盈亏")
    total_return: float = Field(description="总收益率(%)")
    today_profit: float = Field(description="今日盈亏")
    today_return: float = Field(description="今日收益率(%)")


class Position(BaseModel):
    """持仓信息"""
    code: str = Field(description="股票代码")
    name: str = Field(description="股票名称")
    shares: int = Field(description="持仓数量")
    available: int = Field(description="可用数量")
    cost_price: float = Field(description="成本价")
    current_price: float = Field(description="当前价")
    market_value: float = Field(description="市值")
    profit: float = Field(description="盈亏")
    profit_ratio: float = Field(description="盈亏比例(%)")
    weight: float = Field(description="持仓权重(%)")


class AccountOverview(BaseModel):
    """账户完整概览"""
    summary: AccountSummary
    positions: list[Position]
