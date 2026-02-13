"""
Strategy router - 策略API路由
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas.common import ApiResponse
from src.api.schemas.strategy import Strategy, Signal, BacktestConfig
from src.api.services.strategy_service import strategy_service, backtest_service

router = APIRouter(prefix="/api/strategy", tags=["策略"])


@router.get("/list", response_model=ApiResponse[list[Strategy]])
async def get_strategies():
    """获取策略列表"""
    strategies = strategy_service.get_strategies()
    return ApiResponse(data=strategies)


@router.get("/signals", response_model=ApiResponse[list[Signal]])
async def get_signals():
    """获取实时信号"""
    signals = strategy_service.get_signals()
    return ApiResponse(data=signals)


@router.post("/{strategy_id}/start")
async def start_strategy(strategy_id: str):
    """启动策略"""
    success = strategy_service.toggle_strategy(strategy_id, "start")
    if success:
        return {"code": 200, "message": "策略已启动", "data": None}
    raise HTTPException(status_code=404, detail="策略不存在")


@router.post("/{strategy_id}/stop")
async def stop_strategy(strategy_id: str):
    """停止策略"""
    success = strategy_service.toggle_strategy(strategy_id, "stop")
    if success:
        return {"code": 200, "message": "策略已停止", "data": None}
    raise HTTPException(status_code=404, detail="策略不存在")


# ==================== 回测API ====================

@router.post("/backtest/run", response_model=ApiResponse[dict])
async def run_backtest(config: BacktestConfig):
    """启动回测"""
    backtest_id = backtest_service.run_backtest(config)
    return ApiResponse(data={"backtest_id": backtest_id}, message="回测已启动")


@router.get("/backtest/results/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """获取回测结果"""
    result = backtest_service.get_result(backtest_id)
    if result:
        return ApiResponse(data=result)
    raise HTTPException(status_code=404, detail="回测结果不存在")
