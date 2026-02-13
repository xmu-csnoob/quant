"""
ML Router - ML预测API路由
"""

from fastapi import APIRouter, HTTPException, Query
from src.api.schemas.common import ApiResponse
from src.api.schemas.ml import (
    MLPredictionRequest,
    MLPredictionResponse,
    MLModelInfo,
    MLPredictionStats,
    FeatureImportance,
    BatchPredictionRequest,
)
from src.api.services.ml_service import ml_service

router = APIRouter(prefix="/api/ml", tags=["ML预测"])


@router.get("/status")
async def get_ml_status():
    """获取ML模型状态"""
    return {
        "model_loaded": ml_service.is_model_loaded(),
        "model_path": str(ml_service.model_path) if ml_service.model_path else None,
        "feature_count": len(ml_service.feature_cols) if ml_service.feature_cols else 0,
    }


@router.get("/model/info", response_model=ApiResponse[MLModelInfo])
async def get_model_info():
    """获取模型详细信息"""
    info = ml_service.get_model_info()
    if info:
        return ApiResponse(data=info)
    raise HTTPException(status_code=404, detail="模型未加载")


@router.post("/predict", response_model=ApiResponse[MLPredictionResponse])
async def predict_stock(request: MLPredictionRequest):
    """预测单只股票"""
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="ML模型未加载")

    result = ml_service.predict(request.ts_code, request.include_features)
    if result:
        return ApiResponse(data=result)
    raise HTTPException(status_code=404, detail=f"无法预测股票 {request.ts_code}")


@router.post("/predict/batch", response_model=ApiResponse[list[MLPredictionResponse]])
async def batch_predict(request: BatchPredictionRequest):
    """批量预测多只股票"""
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="ML模型未加载")

    results = ml_service.batch_predict(request.ts_codes)
    return ApiResponse(data=results, message=f"成功预测 {len(results)} 只股票")


@router.get("/signals/top", response_model=ApiResponse[list[MLPredictionResponse]])
async def get_top_signals(
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
    signal_type: str = Query("buy", description="信号类型: buy/sell")
):
    """获取TOP信号"""
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="ML模型未加载")

    signals = ml_service.get_top_signals(limit, signal_type)
    return ApiResponse(data=signals)


@router.get("/features/importance", response_model=ApiResponse[list[FeatureImportance]])
async def get_feature_importance(
    top_n: int = Query(20, ge=1, le=50, description="返回数量")
):
    """获取特征重要性"""
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="ML模型未加载")

    importance = ml_service.get_feature_importance(top_n)
    return ApiResponse(data=importance)


@router.get("/stats", response_model=ApiResponse[MLPredictionStats])
async def get_prediction_stats():
    """获取预测统计"""
    stats = ml_service.get_prediction_stats()
    return ApiResponse(data=stats)


@router.get("/{ts_code}", response_model=ApiResponse[MLPredictionResponse])
async def predict_by_code(ts_code: str):
    """根据股票代码预测（GET方式）"""
    if not ml_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="ML模型未加载")

    result = ml_service.predict(ts_code)
    if result:
        return ApiResponse(data=result)
    raise HTTPException(status_code=404, detail=f"无法预测股票 {ts_code}")
