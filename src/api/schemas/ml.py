"""
ML Schema - ML模型相关的数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class PredictionDirection(str, Enum):
    """预测方向"""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class MLPredictionRequest(BaseModel):
    """ML预测请求"""
    ts_code: str = Field(..., description="股票代码，如 600519.SH")
    include_features: bool = Field(False, description="是否包含特征数据")


class MLPredictionResponse(BaseModel):
    """ML预测响应"""
    ts_code: str = Field(..., description="股票代码")
    stock_name: Optional[str] = Field(None, description="股票名称")
    prediction: PredictionDirection = Field(..., description="预测方向")
    probability: float = Field(..., ge=0, le=1, description="上涨概率")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    predicted_return: Optional[float] = Field(None, description="预测收益率")
    signal: str = Field(..., description="交易信号: buy/sell/hold")
    features: Optional[dict] = Field(None, description="特征数据（可选）")
    trade_date: str = Field(..., description="预测日期")
    model_version: str = Field(..., description="模型版本")
    prediction_period: int = Field(5, description="预测周期（天）")


class MLSignal(BaseModel):
    """ML信号"""
    ts_code: str
    stock_name: Optional[str]
    signal_type: str  # buy/sell/hold
    probability: float
    confidence: float
    predicted_return: Optional[float]
    trade_date: str
    created_at: datetime


class MLModelInfo(BaseModel):
    """ML模型信息"""
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(..., description="模型版本")
    model_path: str = Field(..., description="模型路径")
    feature_count: int = Field(..., description="特征数量")
    prediction_period: int = Field(..., description="预测周期")
    train_samples: int = Field(..., description="训练样本数")
    test_samples: int = Field(..., description="测试样本数")
    train_auc: float = Field(..., description="训练集AUC")
    test_auc: float = Field(..., description="测试集AUC")
    train_accuracy: float = Field(..., description="训练集准确率")
    test_accuracy: float = Field(..., description="测试集准确率")
    created_at: str = Field(..., description="创建时间")


class MLPredictionStats(BaseModel):
    """ML预测统计"""
    total_predictions: int = Field(..., description="总预测数")
    correct_predictions: int = Field(..., description="正确预测数")
    accuracy: float = Field(..., description="准确率")
    win_rate: float = Field(..., description="胜率（买入信号）")
    avg_return: float = Field(..., description="平均收益率")
    profit_loss_ratio: float = Field(..., description="盈亏比")
    buy_signals: int = Field(..., description="买入信号数")
    sell_signals: int = Field(..., description="卖出信号数")
    hold_signals: int = Field(..., description="持有信号数")


class FeatureImportance(BaseModel):
    """特征重要性"""
    feature_name: str
    importance_score: float
    rank: int


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    ts_codes: list[str] = Field(..., max_length=100, description="股票代码列表，最多100个")
