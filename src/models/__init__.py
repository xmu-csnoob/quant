"""
模型模块

提供深度学习模型：
- StockLSTMClassifier: LSTM股票涨跌预测模型
"""

from src.models.lstm_model import StockLSTMClassifier

__all__ = [
    "StockLSTMClassifier",
]
