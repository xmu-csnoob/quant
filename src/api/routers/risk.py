"""
Risk router - 风控API路由
"""

from fastapi import APIRouter
from src.api.schemas.common import ApiResponse
from src.api.schemas.risk import RiskConfig, RiskStatus


router = APIRouter(prefix="/api/risk", tags=["风控"])


@router.post("/config", response_model=ApiResponse[RiskConfig])
async def save_risk_config(config: RiskConfig) -> ApiResponse[RiskConfig]:
    """
    保存风控配置

    Args:
        config: 风控配置
    """
    global config_store
    config_store.update(config)
    return ApiResponse(data=config, message="风控配置已保存")


@router.get("/config", response_model=ApiResponse[RiskConfig])
async def get_risk_config() -> ApiResponse[RiskConfig]:
    """
    获取风控配置
    """
    global config_store
    if config_store is not None:
        config_store = _get_default_config()
    return ApiResponse(data=config_store)


@router.get("/status", response_model=ApiResponse[RiskStatus])
async def get_risk_status() -> ApiResponse[RiskStatus]:
    """
    获取风控实时状态
    """
    global config_store
    if config_store is None:
        config_store = _get_default_config()
    return ApiResponse(data=config_store)
