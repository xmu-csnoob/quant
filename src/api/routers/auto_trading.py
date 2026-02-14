"""
Auto Trading Router - 自动交易API路由
"""

from typing import Optional, List
from fastapi import APIRouter, Query
from src.api.schemas.common import ApiResponse
from src.api.services.auto_trading_service import auto_trading_service


router = APIRouter(prefix="/api/auto-trading", tags=["自动交易"])


@router.get("/status", response_model=ApiResponse[dict])
async def get_status():
    """获取自动交易状态"""
    status = auto_trading_service.get_status()
    return ApiResponse(data=status)


@router.post("/run", response_model=ApiResponse[dict])
async def run_trading(
    stock_codes: Optional[str] = Query(None, description="股票代码列表，逗号分隔")
):
    """
    执行自动交易

    分析股票数据，生成信号并执行交易

    Args:
        stock_codes: 可选，指定分析的股票，如 "600000.SH,600519.SH"
    """
    codes = None
    if stock_codes:
        codes = [c.strip() for c in stock_codes.split(",") if c.strip()]

    result = auto_trading_service.analyze_and_trade(codes)
    return ApiResponse(data=result, message=f"交易完成: {len(result['trades'])}笔")


@router.post("/strategy/{strategy_id}", response_model=ApiResponse[dict])
async def set_strategy(strategy_id: str):
    """设置交易策略"""
    success = auto_trading_service.set_strategy(strategy_id)
    if success:
        return ApiResponse(data={"strategy": strategy_id}, message="策略已切换")
    return ApiResponse(code=400, message=f"策略不存在: {strategy_id}")


@router.get("/stocks", response_model=ApiResponse[List[str]])
async def get_available_stocks():
    """获取可用股票列表"""
    stocks = auto_trading_service.get_available_stocks()
    return ApiResponse(data=stocks[:50])  # 返回前50只
