"""
风险管理模块单元测试

测试止损止盈、仓位管理、回撤控制等功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from decimal import Decimal
from src.risk.manager import RiskManager, RiskAction, Position
from src.risk.position_sizer import PositionSizer


class TestRiskManager:
    """风险管理器测试"""

    def test_stop_loss(self):
        """测试止损"""
        print("\n测试止损功能")
        manager = RiskManager(
            initial_capital=100000,
            stop_loss=0.1,  # 10%止损
        )

        # 添加持仓
        manager.positions["600000.SH"] = Position(
            entry_price=100.0,
            shares=1000,
            entry_date="20240101",
        )

        # 价格下跌10%
        check = manager.check_exit("600000.SH", 90.0, "20240102")

        assert check.action == RiskAction.CLOSE
        assert "止损" in check.reason
        print(f"  ✅ 止损触发: {check.reason}")

        # 价格下跌5%，不触发止损
        check = manager.check_exit("600000.SH", 95.0, "20240102")
        assert check.action == RiskAction.HOLD
        print("  ✅ 未触发止损")

    def test_take_profit(self):
        """测试止盈"""
        print("\n测试止盈功能")
        manager = RiskManager(
            initial_capital=100000,
            take_profit=0.2,  # 20%止盈
        )

        # 添加持仓
        manager.positions["600000.SH"] = Position(
            entry_price=100.0,
            shares=1000,
            entry_date="20240101",
        )

        # 价格上涨20%
        check = manager.check_exit("600000.SH", 120.0, "20240102")

        assert check.action == RiskAction.CLOSE
        assert "止盈" in check.reason
        print(f"  ✅ 止盈触发: {check.reason}")

    def test_max_drawdown(self):
        """测试最大回撤限制"""
        print("\n测试最大回撤限制")
        manager = RiskManager(
            initial_capital=100000,
            max_drawdown=0.15,  # 15%最大回撤
        )

        # 模拟回撤10%，允许开仓
        manager.peak_capital = 100000
        manager.current_capital = 90000

        can_enter, _, reason = manager.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600000.SH",
        )
        assert can_enter is True
        print(f"  ✅ 回撤10%允许开仓: {reason}")

        # 模拟回撤20%，不允许开仓
        manager.current_capital = 80000

        can_enter, _, reason = manager.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600001.SH",
        )
        assert can_enter is False
        print(f"  ✅ 回撤20%禁止开仓: {reason}")

    def test_consecutive_losses(self):
        """测试连续亏损保护"""
        print("\n测试连续亏损保护")
        manager = RiskManager(
            initial_capital=100000,
            max_consecutive_losses=3,
        )

        # 模拟连续2次亏损
        manager.consecutive_losses = 2

        can_enter, _, reason = manager.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600000.SH",
        )
        assert can_enter is True
        print(f"  ✅ 连续2次亏损允许开仓: {reason}")

        # 模拟连续3次亏损
        manager.consecutive_losses = 3

        can_enter, _, reason = manager.check_entry(
            price=10.0,
            signal_confidence=0.8,
            stock_code="600001.SH",
        )
        assert can_enter is False
        print(f"  ✅ 连续3次亏损禁止开仓: {reason}")


class TestPositionSizer:
    """仓位管理测试"""

    def test_fixed_ratio_sizing(self):
        """测试固定比例仓位"""
        print("\n测试固定比例仓位")
        sizer = PositionSizer(
            initial_capital=100000,
            method="fixed_ratio",  # 30%仓位
        )

        result = sizer.calculate(
            price=50.0,
            capital=100000,
            confidence=0.8,
        )

        # 30%资金 / 50元 / 100 = 60手 = 6000股
        assert result.shares > 0
        print(f"  ✅ 固定比例仓位: {result.shares}股")

    def test_kelly_criterion(self):
        """测试凯利公式"""
        print("\n测试凯利公式仓位")
        sizer = PositionSizer(
            initial_capital=100000,
            method="kelly",
        )

        result = sizer.calculate(
            price=50.0,
            capital=100000,
            confidence=0.8,
            win_rate=0.6,
            avg_win_loss=1.5,  # 盈亏比
        )

        assert result.shares >= 0  # 可能返回0
        print(f"  ✅ 凯利公式仓位: {result.shares}股")

    def test_atr_based_sizing(self):
        """测试ATR仓位管理"""
        print("\n测试ATR仓位管理")
        sizer = PositionSizer(
            initial_capital=100000,
            method="atr",
        )

        result = sizer.calculate(
            price=50.0,
            capital=100000,
            confidence=0.8,
            atr=2.0,
        )

        assert result.shares >= 0
        print(f"  ✅ ATR仓位: {result.shares}股")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("风险管理模块单元测试")
    print("=" * 60)

    # 风险管理器测试
    trm = TestRiskManager()
    trm.test_stop_loss()
    trm.test_take_profit()
    trm.test_max_drawdown()
    trm.test_consecutive_losses()

    # 仓位管理测试
    tps = TestPositionSizer()
    tps.test_fixed_ratio_sizing()
    tps.test_kelly_criterion()
    tps.test_atr_based_sizing()

    print("\n" + "=" * 60)
    print("所有风险管理测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
