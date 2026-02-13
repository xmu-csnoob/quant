"""
Trading related schemas.
交易相关数据模型
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OrderDirection(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型"""
    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"      # 待提交
    SUBMITTED = "submitted"  # 已提交
    PARTIAL = "partial"      # 部分成交
    FILLED = "filled"        # 全部成交
    CANCELLED = "cancelled"  # 已撤单
    REJECTED = "rejected"    # 已拒绝


class Order(BaseModel):
    """订单信息"""
    order_id: str = Field(description="订单ID")
    code: str = Field(description="股票代码")
    name: str = Field(description="股票名称")
    direction: OrderDirection = Field(description="买卖方向")
    order_type: OrderType = Field(description="订单类型")
    price: Optional[float] = Field(default=None, description="委托价格")
    shares: int = Field(description="委托数量")
    filled_shares: int = Field(default=0, description="成交数量")
    status: OrderStatus = Field(description="订单状态")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")


class CreateOrderRequest(BaseModel):
    """创建订单请求"""
    code: str = Field(description="股票代码")
    direction: OrderDirection = Field(description="买卖方向")
    order_type: OrderType = Field(description="订单类型")
    price: Optional[float] = Field(default=None, description="委托价格(限价单必填)")
    shares: int = Field(gt=0, description="委托数量")
