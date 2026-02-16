"""
ML策略单元测试

测试XGBoost/ML策略的核心功能：初始化、信号生成、阈值逻辑
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from pathlib import Path

from src.strategies.ml_strategy import MLStrategy
from src.strategies.base import SignalType


# ==================== Fixtures ====================

@pytest.fixture
def mock_xgboost_model():
    """创建Mock XGBoost模型"""
    model = Mock()

    def mock_predict(X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.array([0.05, 0.03, -0.02, 0.04, -0.01] * (n_samples // 5 + 1))[:n_samples]

    model.predict = mock_predict
    model.__class__.__name__ = "XGBRegressor"
    return model


@pytest.fixture
def mock_feature_extractor():
    """创建Mock特征提取器"""
    extractor = Mock()

    def mock_extract(df):
        result = df.copy()
        result['f_return_1'] = result['close'].pct_change()
        result['f_return_5'] = result['close'].pct_change(5)
        result['f_ma_ratio'] = result['close'] / result['close'].rolling(20).mean()
        return result

    extractor.extract = mock_extract
    return extractor


@pytest.fixture
def sample_ohlcv():
    """生成样本OHLCV数据"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 10.0
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    return pd.DataFrame({
        'ts_code': '600000.SH',
        'trade_date': dates,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    })


@pytest.fixture
def ml_strategy(mock_xgboost_model, mock_feature_extractor):
    """创建ML策略实例"""
    return MLStrategy(
        model=mock_xgboost_model,
        feature_extractor=mock_feature_extractor,
        threshold=0.02,
        feature_cols=['f_return_1', 'f_return_5', 'f_ma_ratio']
    )


# ==================== 测试类 ====================

class TestMLStrategyInit:
    """测试ML策略初始化"""

    def test_init_success(self, mock_xgboost_model, mock_feature_extractor):
        """测试成功初始化"""
        strategy = MLStrategy(
            model=mock_xgboost_model,
            feature_extractor=mock_feature_extractor,
            threshold=0.02
        )

        assert "XGBRegressor" in strategy.name
        assert strategy.threshold == 0.02

    def test_init_custom_features(self, mock_xgboost_model, mock_feature_extractor):
        """测试自定义特征列"""
        strategy = MLStrategy(
            model=mock_xgboost_model,
            feature_extractor=mock_feature_extractor,
            threshold=0.02,
            feature_cols=['f_1', 'f_2']
        )

        assert strategy.feature_cols == ['f_1', 'f_2']


class TestMLStrategySignals:
    """测试信号生成"""

    def test_generate_signals_buy(self, ml_strategy, sample_ohlcv):
        """测试买入信号生成"""
        def mock_predict_high(X):
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            return np.full(n_samples, 0.05)

        ml_strategy.model.predict = mock_predict_high

        signals = ml_strategy.generate_signals(sample_ohlcv)

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0

    def test_generate_signals_no_signal(self, ml_strategy, sample_ohlcv):
        """测试无信号情况"""
        # 设置高阈值
        strategy = MLStrategy(
            model=ml_strategy.model,
            feature_extractor=ml_strategy.feature_extractor,
            threshold=0.10  # 10%阈值
        )

        def mock_predict_medium(X):
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            return np.full(n_samples, 0.05)

        strategy.model.predict = mock_predict_medium

        signals = strategy.generate_signals(sample_ohlcv)

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) == 0

    def test_insufficient_data(self, ml_strategy):
        """测试数据不足"""
        short_data = pd.DataFrame({
            'trade_date': [pd.Timestamp('2023-01-01')],
            'close': [10.0],
            'volume': [1000000],
        })

        signals = ml_strategy.generate_signals(short_data)
        assert len(signals) == 0


class TestMLStrategyThresholds:
    """测试阈值逻辑"""

    def test_threshold_boundary_buy(self, ml_strategy, sample_ohlcv):
        """测试买入阈值边界"""
        # 预测收益正好在阈值之上
        def mock_predict_boundary(X):
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            return np.full(n_samples, 0.0201)  # 2.01% > 2%阈值

        ml_strategy.model.predict = mock_predict_boundary

        signals = ml_strategy.generate_signals(sample_ohlcv)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]

        assert len(buy_signals) > 0

    def test_negative_prediction(self, ml_strategy, sample_ohlcv):
        """测试负向预测触发卖出"""
        call_count = [0]

        def mock_predict_negative(X):
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            results = []
            for i in range(n_samples):
                call_count[0] += 1
                if call_count[0] < 5:
                    results.append(0.05)  # 先买入
                else:
                    results.append(-0.03)  # 负值触发卖出
            return np.array(results)

        ml_strategy.model.predict = mock_predict_negative

        signals = ml_strategy.generate_signals(sample_ohlcv)
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]

        assert len(sell_signals) > 0


class TestMLStrategyModelIO:
    """测试模型保存和加载"""

    @pytest.mark.skip(reason="待后续PR修复: Mock对象无法pickle")
    def test_save_model(self, ml_strategy, tmp_path):
        """测试模型保存"""
        model_path = tmp_path / "ml_model.pkl"
        ml_strategy.save_model(str(model_path))
        assert model_path.exists()
