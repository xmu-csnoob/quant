"""
Data router - 数据API路由
"""

from datetime import datetime, timedelta
from typing import Optional
import random

from fastapi import APIRouter, Query
from src.api.schemas.common import ApiResponse
from src.api.schemas.strategy import DataStatus, KlineData

router = APIRouter(prefix="/api/data", tags=["数据"])


@router.get("/status", response_model=ApiResponse[DataStatus])
async def get_data_status():
    """获取数据状态"""
    status = DataStatus(
        total_stocks=5234,
        last_update=datetime.now() - timedelta(minutes=15),
        data_sources=["tushare", "akshare"],
        update_status="idle"
    )
    return ApiResponse(data=status)


@router.get("/kline/{code}", response_model=ApiResponse[list[KlineData]])
async def get_kline(
    code: str,
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期")
):
    """获取K线数据"""
    # 生成模拟K线数据
    kline_data = []
    base_price = random.uniform(10, 100)

    start = datetime(2024, 1, 1) if not start_date else datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime(2024, 12, 31) if not end_date else datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start
    price = base_price

    while current_date <= end:
        # 跳过周末
        if current_date.weekday() < 5:
            # 模拟价格波动
            change = random.gauss(0, 0.02)
            open_price = price
            close_price = price * (1 + change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))

            kline_data.append(KlineData(
                date=current_date.strftime("%Y-%m-%d"),
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=random.randint(100000, 10000000),
                amount=random.randint(1000000, 100000000)
            ))

            price = close_price

        current_date += timedelta(days=1)

    return ApiResponse(data=kline_data)
