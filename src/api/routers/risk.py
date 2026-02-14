"""
Risk router - 风控API路由
"""

from fastapi import APIRouter
from src.api.schemas.common import ApiResponse
from src.api.schemas.risk import RiskConfig, RiskStatus
from src.api.services.risk_service import risk_service


router = APIRouter(prefix="/api/risk", tags=["风控"])


@router.post("/config", response_model=ApiResponse[RiskConfig])
async def save_risk_config(config: RiskConfig) -> ApiResponse[RiskConfig]:
    """保存风控配置"""
    saved = risk_service.save_config(config)
    return ApiResponse(data=saved, message="风控配置已保存")


@router.get("/config", response_model=ApiResponse[RiskConfig])
async def get_risk_config() -> ApiResponse[RiskConfig]:
    """获取风控配置"""
    config = risk_service.get_config()
    return ApiResponse(data=config)


@router.get("/status", response_model=ApiResponse[RiskStatus])
async def get_risk_status() -> ApiResponse[RiskStatus]:
    """获取风控实时状态"""
    status = risk_service.get_status()
    return ApiResponse(data=status)
