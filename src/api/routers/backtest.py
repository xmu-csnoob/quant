"""
Backtest router - 回测API路由
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas.common import ApiResponse
from src.api.schemas.strategy import BacktestConfig, BacktestResult
from src.api.services.strategy_service import backtest_service

router = APIRouter(prefix="/api/backtest", tags=["回测"])


@router.post("/run", response_model=ApiResponse[dict])
async def run_backtest(config: BacktestConfig):
    """启动回测"""
    backtest_id = backtest_service.run_backtest(config)
    return ApiResponse(data={"backtest_id": backtest_id}, message="回测已启动")


@router.get("/results/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """获取回测结果"""
    result = backtest_service.get_result(backtest_id)
    if result:
        return ApiResponse(data=result)
    raise HTTPException(status_code=404, detail="回测结果不存在")
