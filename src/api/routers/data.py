"""
Data router - 数据API路由
"""

from datetime import datetime, timedelta
from typing import Optional
import random

from fastapi import APIRouter, Query
from src.api.schemas.common import ApiResponse
from src.api.schemas.strategy import DataStatus, KlineData
from src.data.storage.sqlite_storage import SQLiteStorage

router = APIRouter(prefix="/api/data", tags=["数据"])

# 数据库存储实例
_storage = None


def get_storage():
    """获取数据库存储实例"""
    global _storage
    if _storage is None:
        _storage = SQLiteStorage()
    return _storage


@router.get("/status", response_model=ApiResponse[DataStatus])
async def get_data_status():
    """获取数据状态"""
    storage = get_storage()

    try:
        # 获取真实数据
        all_stocks = storage.get_all_stocks()
        date_range = storage.get_global_date_range()

        if date_range and date_range[0] and date_range[1]:
            min_date = date_range[0]
            max_date = date_range[1]
            total_records = date_range[2] if len(date_range) > 2 else 0
        else:
            min_date = None
            max_date = None
            total_records = 0

        status = DataStatus(
            total_stocks=len(all_stocks),
            last_update=datetime.now(),
            data_sources=["akshare"],
            update_status="idle",
            min_date=min_date,
            max_date=max_date,
            total_records=total_records,
        )
    except Exception as e:
        # 如果获取数据失败，返回默认值
        status = DataStatus(
            total_stocks=0,
            last_update=datetime.now(),
            data_sources=[],
            update_status="error",
            min_date=None,
            max_date=None,
            total_records=0,
        )

    return ApiResponse(data=status)


@router.get("/date-range", response_model=ApiResponse[dict])
async def get_date_range():
    """获取可用日期范围"""
    storage = get_storage()

    try:
        date_range = storage.get_global_date_range()

        if date_range and date_range[0] and date_range[1]:
            return ApiResponse(data={
                "min_date": date_range[0],
                "max_date": date_range[1],
                "total_records": date_range[2] if len(date_range) > 2 else 0,
                "available": True
            })
        else:
            return ApiResponse(data={
                "min_date": None,
                "max_date": None,
                "total_records": 0,
                "available": False
            })
    except Exception as e:
        return ApiResponse(data={
            "min_date": None,
            "max_date": None,
            "total_records": 0,
            "available": False,
            "error": str(e)
        })


@router.get("/kline/{code}", response_model=ApiResponse[list[KlineData]])
async def get_kline(
    code: str,
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期")
):
    """获取K线数据 - 从SQLite数据库读取真实数据"""
    storage = get_storage()

    try:
        # 从数据库获取K线数据
        df = storage.get_daily_prices(code, start_date, end_date)

        if df is None or len(df) == 0:
            return ApiResponse(data=[], message="没有找到数据")

        # 转换为KlineData格式
        kline_data = []
        for _, row in df.iterrows():
            trade_date = row.get('trade_date', '')
            if hasattr(trade_date, 'strftime'):
                date_str = trade_date.strftime("%Y-%m-%d")
            else:
                date_str = str(trade_date)[:10] if len(str(trade_date)) >= 10 else str(trade_date)

            kline_data.append(KlineData(
                date=date_str,
                open=float(row.get('open', 0)),
                high=float(row.get('high', 0)),
                low=float(row.get('low', 0)),
                close=float(row.get('close', 0)),
                volume=float(row.get('vol', row.get('volume', 0))),
                amount=float(row.get('amount', 0))
            ))

        return ApiResponse(data=kline_data)

    except Exception as e:
        return ApiResponse(data=[], message=f"获取数据失败: {str(e)}")
