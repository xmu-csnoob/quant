"""
风控服务单元测试

测试RiskService的核心功能：配置管理、权益计算、风险状态
"""

import pytest
import pandas as pd
from unittest.mock import Mock
from datetime import datetime

from src.api.services.risk_service import RiskService
from src.api.schemas.risk import RiskConfig


# ==================== Fixtures ====================

@pytest.fixture
def mock_storage():
    """创建Mock SQLiteStorage"""
    storage = Mock()

    storage.get_account_balance.return_value = {
        'cash': 1000000,
        'initial_capital': 1000000
    }

    storage.get_positions.return_value = pd.DataFrame({
        'ts_code': ['600000.SH', '000001.SZ'],
        'shares': [1000, 500],
        'market_value': [11000, 9500],
        'unrealized_pnl': [1000, -500]
    })

    storage.get_trades.return_value = pd.DataFrame()

    return storage


@pytest.fixture
def risk_service(mock_storage):
    """创建RiskService实例"""
    service = RiskService()
    service._storage = mock_storage
    return service


# ==================== 测试类 ====================

class TestRiskServiceConfig:
    """测试配置管理"""

    def test_get_default_config(self, risk_service):
        """测试获取默认配置"""
        config = risk_service.get_config()

        assert isinstance(config, RiskConfig)
        assert config.max_position_count == 3
        assert config.max_position_ratio == 30.0
        assert config.stop_loss_ratio == 10.0

    def test_save_config(self, risk_service):
        """测试保存配置"""
        custom_config = RiskConfig(
            max_position_count=5,
            max_position_ratio=50.0
        )

        result = risk_service.save_config(custom_config)

        assert result.max_position_count == 5
        assert result.max_position_ratio == 50.0


class TestRiskServiceEquityCalculation:
    """测试权益计算"""

    def test_get_total_equity_with_positions(self, risk_service, mock_storage):
        """测试有持仓时的总权益计算"""
        total_equity = risk_service._get_total_equity()

        assert total_equity == 1020500.0

    def test_get_total_equity_no_positions(self, risk_service, mock_storage):
        """测试无持仓时的总权益"""
        mock_storage.get_positions.return_value = pd.DataFrame()

        total_equity = risk_service._get_total_equity()

        assert total_equity == 1000000.0

    def test_get_total_equity_error_handling(self, risk_service, mock_storage):
        """测试错误处理"""
        mock_storage.get_account_balance.side_effect = Exception("Database error")

        total_equity = risk_service._get_total_equity()

        assert total_equity == 1000000.0


class TestRiskServiceDailyLossCalculation:
    """测试当日亏损计算"""

    def test_calculate_daily_loss_with_unrealized(self, risk_service):
        """测试有浮亏的情况"""
        daily_loss = risk_service._calculate_daily_loss()

        assert daily_loss >= 0

    def test_calculate_daily_loss_no_loss(self, risk_service, mock_storage):
        """测试无亏损情况"""
        mock_storage.get_positions.return_value = pd.DataFrame({
            'ts_code': ['600000.SH'],
            'shares': [1000],
            'unrealized_pnl': [1000]
        })

        daily_loss = risk_service._calculate_daily_loss()

        assert daily_loss == 0.0

    def test_calculate_daily_loss_error_handling(self, risk_service, mock_storage):
        """测试错误处理"""
        mock_storage.get_trades.side_effect = Exception("Database error")

        daily_loss = risk_service._calculate_daily_loss()

        assert daily_loss == 0.0


class TestRiskServiceGetStatus:
    """测试获取风控状态"""

    def test_get_status_normal(self, risk_service):
        """测试正常状态获取"""
        status = risk_service.get_status()

        assert hasattr(status, 'current_positions')
        assert hasattr(status, 'position_ratio')
        assert hasattr(status, 'daily_loss')
        assert hasattr(status, 'risk_level')
        assert status.risk_level in ["low", "medium", "high"]

    def test_get_status_no_positions(self, risk_service, mock_storage):
        """测试无持仓情况"""
        mock_storage.get_positions.return_value = pd.DataFrame()

        status = risk_service.get_status()

        assert status.current_positions == 0
        assert status.position_ratio == 0.0

    def test_get_status_error_handling(self, risk_service, mock_storage):
        """测试错误处理"""
        mock_storage.get_positions.side_effect = Exception("Database error")

        status = risk_service.get_status()

        assert status.current_positions == 0
        assert status.risk_level == "low"


class TestRiskServiceMaxDrawdown:
    """测试最大回撤计算"""

    def test_calculate_max_drawdown_no_peak(self, risk_service):
        """测试无峰值时的回撤"""
        risk_service._peak_equity = 0

        drawdown = risk_service._calculate_max_drawdown()

        assert drawdown == 0.0

    def test_calculate_max_drawdown_with_peak(self, risk_service, mock_storage):
        """测试有峰值时的回撤"""
        risk_service._peak_equity = 1100000.0

        drawdown = risk_service._calculate_max_drawdown()

        assert drawdown >= 0

    def test_calculate_max_drawdown_error_handling(self, risk_service, mock_storage):
        """测试错误处理"""
        mock_storage.get_account_balance.side_effect = Exception("Database error")

        drawdown = risk_service._calculate_max_drawdown()

        assert drawdown == 0.0
