"""
仓位管理器单元测试

测试PositionSizer的核心功能：固定比例法、凯利公式、ATR止损法
"""

import pytest
import numpy as np
from src.risk.position_sizer import PositionSizer, PositionSize


# ==================== Fixtures ====================

@pytest.fixture
def position_sizer():
    """创建PositionSizer实例"""
    return PositionSizer(
        initial_capital=100000,
        method="fixed_ratio",
        max_position_ratio=0.95,
        min_position_ratio=0.05
    )


@pytest.fixture
def position_sizer_kelly():
    """创建使用凯利公式的PositionSizer"""
    return PositionSizer(
        initial_capital=100000,
        method="kelly",
        max_position_ratio=0.95,
        min_position_ratio=0.05
    )


@pytest.fixture
def position_sizer_atr():
    """创建使用ATR方法的PositionSizer"""
    return PositionSizer(
        initial_capital=100000,
        method="atr",
        max_position_ratio=0.95,
        min_position_ratio=0.05
    )


# ==================== 测试类 ====================

class TestPositionSizerInit:
    """测试初始化"""

    def test_init_default(self):
        """测试默认初始化"""
        sizer = PositionSizer()

        assert sizer.initial_capital == 100000
        assert sizer.method == "fixed_ratio"
        assert sizer.max_position_ratio == 0.95
        assert sizer.min_position_ratio == 0.05

    def test_init_custom(self):
        """测试自定义参数初始化"""
        sizer = PositionSizer(
            initial_capital=500000,
            method="kelly",
            max_position_ratio=0.90,
            min_position_ratio=0.10
        )

        assert sizer.initial_capital == 500000
        assert sizer.method == "kelly"
        assert sizer.max_position_ratio == 0.90


class TestFixedRatioMethod:
    """测试固定比例法"""

    def test_high_confidence(self, position_sizer):
        """测试高置信度（>0.8）-> 90%仓位"""
        result = position_sizer.calculate(
            price=10.0,
            capital=100000,
            confidence=0.85
        )

        expected_ratio = 0.9
        assert abs(result.risk_ratio - expected_ratio) < 0.01 or result.risk_ratio == 0.95

    def test_medium_high_confidence(self, position_sizer):
        """测试中高置信度（0.6-0.8）-> 70%仓位"""
        result = position_sizer.calculate(
            price=10.0,
            capital=100000,
            confidence=0.7
        )

        assert result.risk_ratio == 0.7

    def test_medium_confidence(self, position_sizer):
        """测试中等置信度（0.4-0.6）-> 50%仓位"""
        result = position_sizer.calculate(
            price=10.0,
            capital=100000,
            confidence=0.5
        )

        assert result.risk_ratio == 0.5

    def test_low_confidence(self, position_sizer):
        """测试低置信度（<0.4）-> 30%仓位"""
        result = position_sizer.calculate(
            price=10.0,
            capital=100000,
            confidence=0.3
        )

        assert result.risk_ratio == 0.3

    def test_shares_is_round_lot(self, position_sizer):
        """测试股数是整手（100的倍数）"""
        result = position_sizer.calculate(
            price=13.5,
            capital=100000,
            confidence=0.7
        )

        assert result.shares % 100 == 0


class TestKellyMethod:
    """测试凯利公式"""

    def test_kelly_basic(self, position_sizer_kelly):
        """测试基本凯利公式计算"""
        result = position_sizer_kelly.calculate(
            price=10.0,
            capital=100000,
            win_rate=0.6,
            avg_win_loss=2.0,
            confidence=0.8
        )

        assert isinstance(result, PositionSize)
        assert result.shares > 0
        assert "凯利公式" in result.reason

    def test_kelly_high_win_rate(self, position_sizer_kelly):
        """测试高胜率情况"""
        result = position_sizer_kelly.calculate(
            price=10.0,
            capital=100000,
            win_rate=0.8,
            avg_win_loss=2.0,
            confidence=1.0
        )

        assert result.risk_ratio > 0.1

    def test_kelly_low_win_rate(self, position_sizer_kelly):
        """测试低胜率情况"""
        result = position_sizer_kelly.calculate(
            price=10.0,
            capital=100000,
            win_rate=0.3,
            avg_win_loss=1.0,
            confidence=1.0
        )

        assert result.risk_ratio >= position_sizer_kelly.min_position_ratio

    def test_kelly_default_values(self, position_sizer_kelly):
        """测试默认值（未提供win_rate和avg_win_loss）"""
        result = position_sizer_kelly.calculate(
            price=10.0,
            capital=100000,
            confidence=0.8
        )

        assert isinstance(result, PositionSize)
        assert result.shares > 0


class TestATRMethod:
    """测试ATR止损法"""

    def test_atr_basic(self, position_sizer_atr):
        """测试基本ATR方法"""
        result = position_sizer_atr.calculate(
            price=10.0,
            capital=100000,
            atr=0.5,
            confidence=0.8
        )

        assert isinstance(result, PositionSize)
        assert result.shares > 0
        assert "ATR止损法" in result.reason

    def test_atr_no_atr_value(self, position_sizer_atr):
        """测试未提供ATR值（回退到固定比例）"""
        result = position_sizer_atr.calculate(
            price=10.0,
            capital=100000,
            atr=None,
            confidence=0.7
        )

        assert isinstance(result, PositionSize)
        assert result.risk_ratio == 0.7

    def test_atr_high_volatility(self, position_sizer_atr):
        """测试高波动率（高ATR）"""
        result_high_atr = position_sizer_atr.calculate(
            price=10.0,
            capital=100000,
            atr=2.0,
            confidence=0.8
        )

        result_low_atr = position_sizer_atr.calculate(
            price=10.0,
            capital=100000,
            atr=0.2,
            confidence=0.8
        )

        assert result_high_atr.shares <= result_low_atr.shares


class TestPositionSizerEdgeCases:
    """测试边缘情况"""

    def test_zero_capital(self, position_sizer):
        """测试零资金"""
        result = position_sizer.calculate(
            price=10.0,
            capital=0.0,
            confidence=0.7
        )

        assert result.shares == 0
        assert result.amount == 0.0

    def test_very_low_capital(self, position_sizer):
        """测试资金不足的情况"""
        result = position_sizer.calculate(
            price=10.0,
            capital=50,
            confidence=0.7
        )

        assert result.shares == 0

    def test_unknown_method(self):
        """测试未知方法（应回退到固定比例）"""
        sizer = PositionSizer(
            initial_capital=100000,
            method="unknown_method"
        )

        result = sizer.calculate(
            price=10.0,
            capital=100000,
            confidence=0.7
        )

        assert isinstance(result, PositionSize)
        assert result.shares >= 0

    def test_negative_confidence(self, position_sizer):
        """测试负置信度"""
        result = position_sizer.calculate(
            price=10.0,
            capital=100000,
            confidence=-0.5
        )

        assert result.risk_ratio >= position_sizer.min_position_ratio


class TestPositionSizeDataClass:
    """测试PositionSize数据类"""

    def test_position_size_creation(self):
        """测试PositionSize创建"""
        pos = PositionSize(
            shares=1000,
            amount=10000.0,
            risk_ratio=0.1,
            reason="测试"
        )

        assert pos.shares == 1000
        assert pos.amount == 10000.0
        assert pos.risk_ratio == 0.1
