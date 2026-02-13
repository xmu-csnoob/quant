"""
Trading router - 交易API路由
"""

from typing import Optional
from fastapi import APIRouter, Query
from src.api.schemas.common import ApiResponse, PaginatedResponse
from src.api.schemas.trading import Order, CreateOrderRequest
from src.api.services.trading_service import trading_service

router = APIRouter(prefix="/api/trading", tags=["交易"])


@router.get("/orders", response_model=ApiResponse[PaginatedResponse[Order]])
async def get_orders(
    status: Optional[str] = Query(None, description="订单状态"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """获取订单列表"""
    orders, total = trading_service.get_orders(status, page, page_size)
    return ApiResponse(data=PaginatedResponse(
        items=orders,
        total=total,
        page=page,
        page_size=page_size
    ))


@router.post("/orders", response_model=ApiResponse[Order])
async def create_order(request: CreateOrderRequest):
    """创建订单"""
    order = trading_service.create_order(request)
    return ApiResponse(data=order, message="订单创建成功")


@router.delete("/orders/{order_id}", response_model=ApiResponse[None])
async def cancel_order(order_id: str):
    """取消订单"""
    success = trading_service.cancel_order(order_id)
    if success:
        return ApiResponse(message="订单已取消")
    return ApiResponse(code=400, message="取消订单失败")
