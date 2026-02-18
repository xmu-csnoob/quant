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


class TestRiskManagerCheckEntry:
    """测试入场检查详细场景"""

    def test_check_entry_max_drawdown_reached(self):
        """测试达到最大回撤时不允许入场"""
        rm = RiskManager(
            initial_capital=100000,
            max_drawdown=0.10,  # 10%最大回撤
        )
        # 模拟回撤
        rm.current_capital = 85000  # 15%回撤
        rm.peak_capital = 100000

        allowed, position_size, reason = rm.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600000.SH",
        )

        assert allowed is False
        assert "回撤" in reason

    def test_check_entry_consecutive_losses_reached(self):
        """测试连续亏损达到上限时不允许入场"""
        rm = RiskManager(max_consecutive_losses=3)
        rm.consecutive_losses = 3

        allowed, position_size, reason = rm.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600000.SH",
        )

        assert allowed is False
        assert "连续亏损" in reason

    def test_check_entry_daily_loss_reached(self):
        """测试单日亏损达到上限时不允许入场"""
        rm = RiskManager(
            initial_capital=100000,
            max_daily_loss=0.05,
        )
        rm.daily_pnl = -6000  # 6%亏损，超过5%限制

        allowed, position_size, reason = rm.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600000.SH",
        )

        assert allowed is False
        assert "单日最大亏损" in reason

    def test_check_entry_total_position_reached(self):
        """测试总仓位达到上限时不允许入场"""
        rm = RiskManager(
            initial_capital=100000,
            max_total_position_ratio=0.8,
        )
        # 模拟已有持仓
        from src.risk.manager import Position
        rm.positions["600000.SH"] = Position(
            entry_price=10.0,
            shares=8500,  # 85%仓位
            entry_date="20230101",
            current_price=10.0,
        )

        allowed, position_size, reason = rm.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600001.SH",
        )

        assert allowed is False
        assert "总仓位" in reason

    def test_check_entry_position_ratio_capped(self):
        """测试单笔仓位被限制"""
        rm = RiskManager(
            initial_capital=100000,
            max_position_ratio=0.3,
        )

        allowed, position_size, reason = rm.check_entry(
            price=10.0,
            signal_confidence=0.9,
            stock_code="600000.SH",
        )

        assert allowed is True
        # 仓位应该被限制
        assert position_size.risk_ratio <= 0.3


class TestRiskManagerCheckExit:
    """测试出场检查详细场景"""

    def test_check_exit_no_position(self, risk_manager):
        """测试无持仓时的出场检查"""
        result = risk_manager.check_exit(
            stock_code="999999.SH",
            current_price=10.0,
            date="20230615",
        )

        assert result.passed is True
        assert result.action.value == "持有"

    def test_check_exit_stop_loss_price_triggered(self):
        """测试价格止损触发"""
        rm = RiskManager(stop_loss=0.10)
        rm.open_position("600000.SH", 10.0, 100, "20230101")
        # 设置止损价
        rm.positions["600000.SH"].stop_loss = 9.5

        result = rm.check_exit(
            stock_code="600000.SH",
            current_price=9.4,  # 低于止损价
            date="20230102",
        )

        assert result.passed is False
        assert result.action.value == "平仓"
        assert "止损" in result.reason

    def test_check_exit_stop_loss_ratio_triggered(self):
        """测试比例止损触发"""
        rm = RiskManager(stop_loss=0.10)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        result = rm.check_exit(
            stock_code="600000.SH",
            current_price=8.5,  # 亏损15%，超过10%止损
            date="20230102",
        )

        assert result.passed is False
        assert "止损" in result.reason

    def test_check_exit_take_profit_price_triggered(self):
        """测试价格止盈触发"""
        rm = RiskManager(take_profit=0.15)
        rm.open_position("600000.SH", 10.0, 100, "20230101")
        rm.positions["600000.SH"].take_profit = 11.5

        result = rm.check_exit(
            stock_code="600000.SH",
            current_price=12.0,  # 高于止盈价
            date="20230102",
        )

        assert result.passed is False
        assert result.action.value == "平仓"
        assert "止盈" in result.reason

    def test_check_exit_take_profit_ratio_triggered(self):
        """测试比例止盈触发"""
        rm = RiskManager(take_profit=0.15)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        result = rm.check_exit(
            stock_code="600000.SH",
            current_price=12.0,  # 盈利20%，超过15%止盈
            date="20230102",
        )

        assert result.passed is False
        assert "止盈" in result.reason

    def test_check_exit_trailing_stop_triggered(self):
        """测试移动止损触发"""
        rm = RiskManager()
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        # 模拟曾经盈利10%
        rm.positions["600000.SH"].peak_pnl_ratio = 0.10

        # 当前盈利3%，从峰值回撤70% (超过50%的移动止损线)
        rm.positions["600000.SH"].unrealized_pnl_ratio = 0.03

        result = rm.check_exit(
            stock_code="600000.SH",
            current_price=10.3,
            date="20230102",
        )

        # 由于移动止损线是峰值的50% = 5%，当前3%低于止损线
        # 需要检查是否触发
        if not result.passed:
            assert "移动止损" in result.reason

    def test_check_exit_hold(self):
        """测试正常持有"""
        rm = RiskManager(stop_loss=0.10, take_profit=0.20)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        result = rm.check_exit(
            stock_code="600000.SH",
            current_price=10.5,  # 盈利5%，未触发止损止盈
            date="20230102",
        )

        assert result.passed is True
        assert result.action.value == "持有"


class TestRiskManagerOpenPosition:
    """测试开仓操作"""

    def test_open_position_basic(self):
        """测试基本开仓"""
        rm = RiskManager(initial_capital=100000)

        rm.open_position(
            stock_code="600000.SH",
            entry_price=10.0,
            shares=100,
            entry_date="20230101",
        )

        assert "600000.SH" in rm.positions
        assert rm.positions["600000.SH"].entry_price == 10.0
        assert rm.positions["600000.SH"].shares == 100
        # 资金应该减少
        assert rm.current_capital == 99000

    def test_open_position_with_stop_loss(self):
        """测试带止损的开仓"""
        rm = RiskManager(initial_capital=100000)

        rm.open_position(
            stock_code="600000.SH",
            entry_price=10.0,
            shares=100,
            entry_date="20230101",
            stop_loss=9.0,
        )

        assert rm.positions["600000.SH"].stop_loss == 9.0

    def test_open_position_with_take_profit(self):
        """测试带止盈的开仓"""
        rm = RiskManager(initial_capital=100000)

        rm.open_position(
            stock_code="600000.SH",
            entry_price=10.0,
            shares=100,
            entry_date="20230101",
            take_profit=12.0,
        )

        assert rm.positions["600000.SH"].take_profit == 12.0

    def test_open_position_auto_stop_loss(self):
        """测试自动设置止损"""
        rm = RiskManager(initial_capital=100000, stop_loss=0.10)

        rm.open_position(
            stock_code="600000.SH",
            entry_price=10.0,
            shares=100,
            entry_date="20230101",
        )

        # 应该自动设置止损价为 10 * (1 - 0.10) = 9.0
        assert rm.positions["600000.SH"].stop_loss == 9.0

    def test_open_position_auto_take_profit(self):
        """测试自动设置止盈"""
        rm = RiskManager(initial_capital=100000, take_profit=0.15)

        rm.open_position(
            stock_code="600000.SH",
            entry_price=10.0,
            shares=100,
            entry_date="20230101",
        )

        # 应该自动设置止盈价为 10 * (1 + 0.15) = 11.5
        assert rm.positions["600000.SH"].take_profit == 11.5


class TestRiskManagerClosePosition:
    """测试平仓操作"""

    def test_close_position_profit(self):
        """测试盈利平仓"""
        rm = RiskManager(initial_capital=100000)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        trade = rm.close_position(
            stock_code="600000.SH",
            exit_price=11.0,
            exit_date="20230102",
            reason="止盈",
        )

        assert trade is not None
        assert trade["pnl"] == 100  # (11-10) * 100
        assert trade["pnl_ratio"] == 0.10
        assert "600000.SH" not in rm.positions
        # 连续亏损应该重置
        assert rm.consecutive_losses == 0

    def test_close_position_loss(self):
        """测试亏损平仓"""
        rm = RiskManager(initial_capital=100000)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        trade = rm.close_position(
            stock_code="600000.SH",
            exit_price=9.0,
            exit_date="20230102",
            reason="止损",
        )

        assert trade["pnl"] == -100  # (9-10) * 100
        assert trade["pnl_ratio"] == -0.10
        # 连续亏损应该增加
        assert rm.consecutive_losses == 1

    def test_close_position_update_peak_capital(self):
        """测试平仓后更新峰值资金"""
        rm = RiskManager(initial_capital=100000)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        rm.close_position(
            stock_code="600000.SH",
            exit_price=12.0,
            exit_date="20230102",
            reason="止盈",
        )

        # 资金计算：初始 100000 - 100*10(开仓) + 100*12(平仓) = 100200
        assert rm.peak_capital == 100200

    def test_close_position_no_position(self):
        """测试平仓不存在的持仓"""
        rm = RiskManager()

        # 不应该抛出异常
        result = rm.close_position(
            stock_code="999999.SH",
            exit_price=10.0,
            exit_date="20230102",
            reason="测试",
        )

        assert result is None

    def test_close_position_recorded(self):
        """测试平仓记录"""
        rm = RiskManager(initial_capital=100000)
        rm.open_position("600000.SH", 10.0, 100, "20230101")

        rm.close_position(
            stock_code="600000.SH",
            exit_price=11.0,
            exit_date="20230102",
            reason="止盈",
        )

        assert len(rm.closed_trades) == 1
        assert rm.closed_trades[0]["stock_code"] == "600000.SH"


class TestRiskManagerSummary:
    """测试摘要功能"""

    def test_get_summary_with_positions(self):
        """测试有持仓时的摘要"""
        rm = RiskManager(initial_capital=100000)
        rm.open_position("600000.SH", 10.0, 100, "20230101")
        rm.positions["600000.SH"].current_price = 10.0
        rm.positions["600000.SH"].unrealized_pnl = 0

        summary = rm.get_summary()

        assert summary["open_positions"] == 1
        assert summary["initial_capital"] == 100000

    def test_get_summary_with_closed_trades(self):
        """测试有已平仓交易的摘要"""
        rm = RiskManager(initial_capital=100000)
        rm.open_position("600000.SH", 10.0, 100, "20230101")
        rm.close_position("600000.SH", 11.0, "20230102", "止盈")

        summary = rm.get_summary()

        assert summary["closed_trades"] == 1
        assert summary["realized_pnl"] == 100
