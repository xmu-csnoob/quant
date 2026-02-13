"""
Account router - 账户API路由
"""

from fastapi import APIRouter
from src.api.schemas.common import ApiResponse
from src.api.schemas.account import AccountSummary, Position
from src.api.services.account_service import account_service

router = APIRouter(prefix="/api/account", tags=["账户"])


@router.get("/summary", response_model=ApiResponse[AccountSummary])
async def get_account_summary():
    """获取账户概览"""
    summary = account_service.get_summary()
    return ApiResponse(data=summary)


@router.get("/positions", response_model=ApiResponse[list[Position]])
async def get_positions():
    """获取持仓列表"""
    positions = account_service.get_positions()
    return ApiResponse(data=positions)
