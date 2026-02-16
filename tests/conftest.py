"""
全局pytest配置和fixtures

提供测试所需的通用fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== 样本数据Fixtures ====================

@pytest.fixture
def sample_ohlcv_data():
    """生成样本OHLCV数据（日线）"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 10.0
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    data = pd.DataFrame({
        'ts_code': '600000.SH',
        'trade_date': dates,
        'open': prices * (1 + np.random.randn(100) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.015),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.015),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
    })

    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_data_manager():
    """Mock DataManager"""
    mock_manager = Mock()

    def mock_get_daily_price(ts_code, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(ts_code) % 2**32)
        base_price = 10.0
        returns = np.random.randn(len(dates)) * 0.02
        prices = base_price * (1 + returns).cumprod()

        return pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        })

    mock_manager.get_daily_price = mock_get_daily_price
    mock_manager.get_latest_price = Mock(return_value=10.5)

    return mock_manager


@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager"""
    mock_manager = Mock()
    mock_manager.check_position_limit = Mock(return_value=True)
    mock_manager.check_daily_loss_limit = Mock(return_value=True)
    mock_manager.calculate_position_size = Mock(return_value=1000)

    return mock_manager


@pytest.fixture
def mock_trading_api():
    """Mock TradingAPI"""
    mock_api = Mock()
    mock_api.buy = Mock(return_value={'order_id': '12345', 'status': 'submitted'})
    mock_api.sell = Mock(return_value={'order_id': '12346', 'status': 'submitted'})
    mock_api.get_positions = Mock(return_value={})
    mock_api.get_cash = Mock(return_value=100000.0)

    return mock_api


# ==================== Pytest配置钩子 ====================

def pytest_configure(config):
    """pytest配置钩子 - 注册自定义标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
