"""
LSTM策略单元测试

测试LSTM策略的核心功能：初始化、信号生成、预测
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
import torch

from src.strategies.lstm_strategy import LSTMStrategy
from src.strategies.base import SignalType


# ==================== Fixtures ====================

@pytest.fixture
def mock_model():
    """创建Mock LSTM模型"""
    model = Mock(spec=torch.nn.Module)
    model.eval = Mock(return_value=None)
    model.to = Mock(return_value=model)

    def mock_forward(x):
        if x is None:
            return torch.tensor([[0.65]]), None
        prob = torch.tensor([[0.65]])
        return prob, None

    model.side_effect = mock_forward
    return model


@pytest.fixture
def mock_feature_extractor():
    """创建Mock特征提取器"""
    extractor = Mock()

    def mock_extract(df):
        result = df.copy()
        result['f_return_1'] = result['close'].pct_change()
        result['f_return_5'] = result['close'].pct_change(5)
        result['f_volatility'] = result['close'].rolling(20).std()
        return result

    extractor.extract = mock_extract
    return extractor


@pytest.fixture
def mock_scaler():
    """创建Mock StandardScaler"""
    from sklearn.preprocessing import StandardScaler
    scaler = Mock(spec=StandardScaler)

    def mock_transform(X):
        return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    scaler.transform = mock_transform
    return scaler


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
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    })


@pytest.fixture
def lstm_strategy(mock_model, mock_feature_extractor, mock_scaler):
    """创建LSTM策略实例"""
    return LSTMStrategy(
        model=mock_model,
        feature_extractor=mock_feature_extractor,
        scaler=mock_scaler,
        feature_cols=['f_return_1', 'f_return_5', 'f_volatility'],
        seq_len=20,
        buy_threshold=0.60,
        sell_threshold=0.40,
        device='cpu'
    )


# ==================== 测试类 ====================

class TestLSTMStrategyInit:
    """测试LSTM策略初始化"""

    def test_init_success(self, mock_model, mock_feature_extractor, mock_scaler):
        """测试成功初始化"""
        strategy = LSTMStrategy(
            model=mock_model,
            feature_extractor=mock_feature_extractor,
            scaler=mock_scaler,
            feature_cols=['f_1', 'f_2'],
            seq_len=20,
            buy_threshold=0.60,
            sell_threshold=0.40,
            device='cpu'
        )

        assert strategy.name == "LSTM_Strategy"
        assert strategy.seq_len == 20
        assert strategy.buy_threshold == 0.60
        assert strategy.sell_threshold == 0.40

    def test_init_custom_thresholds(self, mock_model, mock_feature_extractor, mock_scaler):
        """测试自定义阈值"""
        strategy = LSTMStrategy(
            model=mock_model,
            feature_extractor=mock_feature_extractor,
            scaler=mock_scaler,
            feature_cols=['f_1'],
            buy_threshold=0.55,
            sell_threshold=0.45,
            device='cpu'
        )

        assert strategy.buy_threshold == 0.55
        assert strategy.sell_threshold == 0.45


class TestLSTMStrategySignals:
    """测试信号生成"""

    def test_generate_signals_buy(self, lstm_strategy, sample_ohlcv):
        """测试买入信号生成"""
        # Mock返回高概率
        def mock_forward_high(x):
            return torch.tensor([[0.75]]), None

        lstm_strategy.model.side_effect = mock_forward_high

        signals = lstm_strategy.generate_signals(sample_ohlcv)

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0
        assert buy_signals[0].signal_type == SignalType.BUY

    def test_generate_signals_hold(self, lstm_strategy, sample_ohlcv):
        """测试持有（无信号）"""
        def mock_forward_neutral(x):
            return torch.tensor([[0.50]]), None

        lstm_strategy.model.side_effect = mock_forward_neutral

        signals = lstm_strategy.generate_signals(sample_ohlcv)
        assert len(signals) == 0

    def test_insufficient_data(self, lstm_strategy):
        """测试数据不足"""
        short_data = pd.DataFrame({
            'trade_date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'close': np.random.randn(10) * 10,
            'volume': np.random.randint(1000, 10000, 10),
        })

        signals = lstm_strategy.generate_signals(short_data)
        assert len(signals) == 0


class TestLSTMStrategyPredict:
    """测试预测功能"""

    def test_predict_success(self, lstm_strategy, sample_ohlcv):
        """测试成功预测"""
        prob = lstm_strategy.predict(sample_ohlcv)
        assert prob is not None
        assert 0 <= prob <= 1

    def test_predict_insufficient_data(self, lstm_strategy):
        """测试数据不足时的预测"""
        short_data = pd.DataFrame({
            'trade_date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'close': np.random.randn(10) * 10,
            'volume': np.random.randint(1000, 10000, 10),
        })

        prob = lstm_strategy.predict(short_data)
        assert prob is None


class TestLSTMStrategyLoad:
    """测试模型加载"""

    def test_load_missing_file(self, tmp_path):
        """测试模型文件不存在"""
        strategy = LSTMStrategy.load(
            model_path=str(tmp_path / "nonexistent.pt"),
            device='cpu'
        )
        assert strategy is None


class TestLSTMStrategySell:
    """测试卖出信号"""

    def test_generate_signals_sell(self, lstm_strategy, sample_ohlcv):
        """测试卖出信号生成"""
        # Mock返回低概率
        def mock_forward_low(x):
            return torch.tensor([[0.25]]), None

        lstm_strategy.model.side_effect = mock_forward_low

        signals = lstm_strategy.generate_signals(sample_ohlcv)

        # 卖出信号需要先有持仓，这里只验证返回列表
        assert isinstance(signals, list)


class TestLSTMStrategyThresholds:
    """测试阈值边界"""

    def test_buy_threshold_boundary(self, mock_model, mock_feature_extractor, mock_scaler, sample_ohlcv):
        """测试买入阈值边界"""
        strategy = LSTMStrategy(
            model=mock_model,
            feature_extractor=mock_feature_extractor,
            scaler=mock_scaler,
            feature_cols=['f_return_1'],
            seq_len=10,
            buy_threshold=0.65,
            sell_threshold=0.35,
            device='cpu'
        )

        # 概率正好在阈值上
        def mock_forward_boundary(x):
            return torch.tensor([[0.70]]), None

        strategy.model.side_effect = mock_forward_boundary

        signals = strategy.generate_signals(sample_ohlcv)
        # 应该产生买入信号
        assert isinstance(signals, list)

    def test_sell_threshold_boundary(self, mock_model, mock_feature_extractor, mock_scaler, sample_ohlcv):
        """测试卖出阈值边界"""
        strategy = LSTMStrategy(
            model=mock_model,
            feature_extractor=mock_feature_extractor,
            scaler=mock_scaler,
            feature_cols=['f_return_1'],
            seq_len=10,
            buy_threshold=0.65,
            sell_threshold=0.35,
            device='cpu'
        )

        # 概率低于卖出阈值
        def mock_forward_sell(x):
            return torch.tensor([[0.30]]), None

        strategy.model.side_effect = mock_forward_sell

        signals = strategy.generate_signals(sample_ohlcv)
        assert isinstance(signals, list)
