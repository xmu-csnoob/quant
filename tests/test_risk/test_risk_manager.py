"""
风控管理器单元测试

测试RiskManager的核心功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.risk.manager import RiskManager


@pytest.fixture
def risk_manager():
    """创建风控管理器"""
    return RiskManager(
        initial_capital=100000,
        stop_loss=0.10,
        take_profit=0.15,
        max_position_ratio=0.3,
        max_drawdown=0.15,
        max_daily_loss=0.05,
    )


class TestRiskManagerInit:
    """测试初始化"""

    def test_default_params(self):
        """测试默认参数"""
        rm = RiskManager()

        assert rm.max_position_ratio > 0
        assert rm.max_daily_loss > 0

    def test_custom_params(self):
        """测试自定义参数"""
        rm = RiskManager(
            initial_capital=200000,
            max_position_ratio=0.2,
            max_daily_loss=0.03,
        )

        assert rm.initial_capital == 200000
        assert rm.max_position_ratio == 0.2
        assert rm.max_daily_loss == 0.03


class TestRiskManagerChecks:
    """测试风控检查"""

    def test_check_entry_allowed(self, risk_manager):
        """测试允许入场"""
        # 模拟正常的入场检查
        result = risk_manager.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600000.SH",
        )

        # 应该返回检查结果
        assert result is not None

    def test_check_exit_allowed(self, risk_manager):
        """测试允许出场"""
        result = risk_manager.check_exit(
            stock_code="600000.SH",
            current_price=11.0,
            date="20230615",
        )

        # 应该返回检查结果
        assert result is not None


class TestRiskManagerPosition:
    """测试仓位管理"""

    def test_positions_dict_exists(self, risk_manager):
        """测试持仓字典存在"""
        assert hasattr(risk_manager, 'positions')
        assert isinstance(risk_manager.positions, dict)


class TestRiskManagerStats:
    """测试统计功能"""

    def test_get_summary(self, risk_manager):
        """测试获取摘要"""
        summary = risk_manager.get_summary()

        assert isinstance(summary, dict)

    def test_update_daily_pnl(self, risk_manager):
        """测试更新每日盈亏"""
        risk_manager.update_daily_pnl(1000)

        assert risk_manager.daily_pnl == 1000


class TestRiskManagerDailyLoss:
    """测试每日亏损限制"""

    def test_within_daily_limit(self, risk_manager):
        """测试在每日亏损限制内"""
        # 小额亏损
        risk_manager.update_daily_pnl(-1000)

        # 应该允许继续交易
        assert risk_manager.daily_pnl == -1000

    def test_daily_loss_tracking(self, risk_manager):
        """测试每日亏损追踪"""
        # 记录亏损
        risk_manager.update_daily_pnl(-1000)
        risk_manager.update_daily_pnl(-500)

        summary = risk_manager.get_summary()
        assert isinstance(summary, dict)


class TestRiskManagerDrawdown:
    """测试回撤控制"""

    def test_drawdown_tracking(self, risk_manager):
        """测试回撤追踪"""
        # 初始状态
        assert risk_manager.peak_capital == 100000

        # 获取摘要
        summary = risk_manager.get_summary()
        assert isinstance(summary, dict)
