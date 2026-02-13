"""
交易成本计算模块

实现 A 股交易成本模型，包括：
- 佣金（双向）
- 印花税（仅卖出）
- 过户费（双向，仅上海市场）

市场规则：
- 上海市场（SSE）：600/601/603/688 开头，收取过户费
- 深圳市场（SZSE）：000/001/002/003/300/301 开头，不收取过户费
- 北京市场（BSE）：8 开头，不收取过户费（2023年起）
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, Tuple


class Market(Enum):
    """股票市场"""
    SSE = "sse"  # 上海证券交易所
    SZSE = "szse"  # 深圳证券交易所
    BSE = "bse"  # 北京证券交易所
    UNKNOWN = "unknown"  # 未知市场


class TradeSide(Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class CostConfig:
    """
    交易成本配置

    默认值基于 A 股市场常见费率：
    - 佣金：万2.5，最低5元
    - 印花税：千1（仅卖出）
    - 过户费：万0.1（双向，仅上海市场）

    Attributes:
        commission_rate: 佣金费率（默认万2.5）
        min_commission: 最低佣金（默认5元）
        stamp_duty_rate: 印花税率（默认千1）
        transfer_fee_rate: 过户费率（默认万0.1）
    """

    commission_rate: Decimal = Decimal("0.00025")  # 万2.5
    min_commission: Decimal = Decimal("5")  # 最低5元
    stamp_duty_rate: Decimal = Decimal("0.001")  # 千1
    transfer_fee_rate: Decimal = Decimal("0.00001")  # 万0.1

    @classmethod
    def default(cls) -> "CostConfig":
        """获取默认配置"""
        return cls()

    @classmethod
    def no_cost(cls) -> "CostConfig":
        """获取无成本配置（用于对比测试）"""
        return cls(
            commission_rate=Decimal("0"),
            min_commission=Decimal("0"),
            stamp_duty_rate=Decimal("0"),
            transfer_fee_rate=Decimal("0"),
        )

    @classmethod
    def high_cost(cls) -> "CostConfig":
        """获取高成本配置（保守估计）"""
        return cls(
            commission_rate=Decimal("0.0005"),  # 万5
            min_commission=Decimal("5"),
            stamp_duty_rate=Decimal("0.001"),  # 千1（固定）
            transfer_fee_rate=Decimal("0.00002"),  # 万0.2
        )


@dataclass
class CostBreakdown:
    """
    成本明细

    Attributes:
        commission: 佣金
        stamp_duty: 印花税
        transfer_fee: 过户费
        total: 总成本
    """

    commission: Decimal
    stamp_duty: Decimal
    transfer_fee: Decimal
    total: Decimal

    def __str__(self) -> str:
        return (
            f"CostBreakdown(佣金={self.commission:.2f}, "
            f"印花税={self.stamp_duty:.2f}, "
            f"过户费={self.transfer_fee:.2f}, "
            f"合计={self.total:.2f})"
        )


class TransactionCostCalculator:
    """
    交易成本计算器

    根据 A 股交易规则计算交易成本：
    - 买入：佣金 + 过户费
    - 卖出：佣金 + 印花税 + 过户费

    Example:
        >>> config = CostConfig.default()
        >>> calculator = TransactionCostCalculator(config)
        >>> cost = calculator.calculate(TradeSide.BUY, 100000)
        >>> print(cost.total)
    """

    def __init__(self, config: Optional[CostConfig] = None):
        """
        初始化成本计算器

        Args:
            config: 成本配置，默认使用 CostConfig.default()
        """
        self.config = config or CostConfig.default()

    def calculate(self, side: TradeSide, amount: Decimal) -> CostBreakdown:
        """
        计算交易成本

        Args:
            side: 交易方向（买入/卖出）
            amount: 交易金额（价格 × 数量）

        Returns:
            成本明细
        """
        # 佣金：取 max(金额 * 费率, 最低佣金)
        commission = max(
            amount * self.config.commission_rate,
            self.config.min_commission
        )

        # 印花税：仅卖出时收取
        stamp_duty = Decimal("0")
        if side == TradeSide.SELL:
            stamp_duty = amount * self.config.stamp_duty_rate

        # 过户费：双向收取
        transfer_fee = amount * self.config.transfer_fee_rate

        total = commission + stamp_duty + transfer_fee

        return CostBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total=total,
        )

    def calculate_buy_cost(self, amount: Decimal) -> CostBreakdown:
        """
        计算买入成本

        Args:
            amount: 买入金额

        Returns:
            成本明细
        """
        return self.calculate(TradeSide.BUY, amount)

    def calculate_sell_cost(self, amount: Decimal) -> CostBreakdown:
        """
        计算卖出成本

        Args:
            amount: 卖出金额

        Returns:
            成本明细
        """
        return self.calculate(TradeSide.SELL, amount)

    def get_effective_buy_rate(self) -> Decimal:
        """
        获取买入有效费率（不含最低佣金影响）

        Returns:
            买入有效费率
        """
        return self.config.commission_rate + self.config.transfer_fee_rate

    def get_effective_sell_rate(self) -> Decimal:
        """
        获取卖出有效费率（不含最低佣金影响）

        Returns:
            卖出有效费率
        """
        return (
            self.config.commission_rate
            + self.config.stamp_duty_rate
            + self.config.transfer_fee_rate
        )

    def estimate_round_trip_cost(self, amount: Decimal) -> Decimal:
        """
        估算往返交易成本（买入 + 卖出）

        注意：此方法不区分市场，对深圳股票会高估成本。
        建议使用 estimate_round_trip_cost_with_code 方法。

        Args:
            amount: 交易金额

        Returns:
            往返总成本
        """
        buy_cost = self.calculate_buy_cost(amount)
        sell_cost = self.calculate_sell_cost(amount)
        return buy_cost.total + sell_cost.total

    @staticmethod
    def get_market(code: str) -> Market:
        """
        根据股票代码判断所属市场

        Args:
            code: 股票代码，支持格式：'600000', '600000.SH', 'sh600000'

        Returns:
            市场类型
        """
        # 标准化代码：去除市场后缀和前缀
        pure_code = code.upper().replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
        pure_code = pure_code.replace('SH', '').replace('SZ', '').replace('BJ', '')

        # 上海市场
        if pure_code.startswith(('600', '601', '603', '605', '688', '689')):
            return Market.SSE
        # 深圳市场
        elif pure_code.startswith(('000', '001', '002', '003', '300', '301')):
            return Market.SZSE
        # 北京市场
        elif pure_code.startswith(('8', '4')):
            return Market.BSE
        else:
            return Market.UNKNOWN

    @staticmethod
    def is_shanghai_market(code: str) -> bool:
        """
        判断是否为上海市场股票

        Args:
            code: 股票代码

        Returns:
            是否为上海市场
        """
        return TransactionCostCalculator.get_market(code) == Market.SSE

    def _calculate_transfer_fee(self, code: str, amount: Decimal) -> Decimal:
        """
        根据市场计算过户费

        规则：
        - 上海市场（SSE）：收取过户费，费率万0.1
        - 深圳市场（SZSE）：不收取过户费
        - 北京市场（BSE）：不收取过户费

        Args:
            code: 股票代码
            amount: 交易金额

        Returns:
            过户费金额
        """
        market = self.get_market(code)
        if market == Market.SSE:
            return amount * self.config.transfer_fee_rate
        # 深圳和北京市场不收取过户费
        return Decimal("0")

    def calculate_with_code(
        self, side: TradeSide, code: str, amount: Decimal
    ) -> CostBreakdown:
        """
        计算交易成本（根据股票代码区分市场）

        Args:
            side: 交易方向（买入/卖出）
            code: 股票代码
            amount: 交易金额（价格 × 数量）

        Returns:
            成本明细
        """
        # 佣金：取 max(金额 * 费率, 最低佣金)
        commission = max(
            amount * self.config.commission_rate,
            self.config.min_commission
        )

        # 印花税：仅卖出时收取
        stamp_duty = Decimal("0")
        if side == TradeSide.SELL:
            stamp_duty = amount * self.config.stamp_duty_rate

        # 过户费：仅上海市场收取
        transfer_fee = self._calculate_transfer_fee(code, amount)

        total = commission + stamp_duty + transfer_fee

        return CostBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total=total,
        )

    def calculate_buy_cost_with_code(
        self, code: str, amount: Decimal
    ) -> CostBreakdown:
        """
        计算买入成本（根据股票代码区分市场）

        Args:
            code: 股票代码
            amount: 买入金额

        Returns:
            成本明细
        """
        return self.calculate_with_code(TradeSide.BUY, code, amount)

    def calculate_sell_cost_with_code(
        self, code: str, amount: Decimal
    ) -> CostBreakdown:
        """
        计算卖出成本（根据股票代码区分市场）

        Args:
            code: 股票代码
            amount: 卖出金额

        Returns:
            成本明细
        """
        return self.calculate_with_code(TradeSide.SELL, code, amount)

    def get_effective_buy_rate_with_code(self, code: str) -> Decimal:
        """
        获取指定股票的买入有效费率（不含最低佣金影响）

        Args:
            code: 股票代码

        Returns:
            买入有效费率
        """
        rate = self.config.commission_rate
        if self.is_shanghai_market(code):
            rate += self.config.transfer_fee_rate
        return rate

    def get_effective_sell_rate_with_code(self, code: str) -> Decimal:
        """
        获取指定股票的卖出有效费率（不含最低佣金影响）

        Args:
            code: 股票代码

        Returns:
            卖出有效费率
        """
        rate = (
            self.config.commission_rate
            + self.config.stamp_duty_rate
        )
        if self.is_shanghai_market(code):
            rate += self.config.transfer_fee_rate
        return rate

    def estimate_round_trip_cost_with_code(
        self, code: str, amount: Decimal
    ) -> Decimal:
        """
        估算指定股票的往返交易成本（买入 + 卖出）

        Args:
            code: 股票代码
            amount: 交易金额

        Returns:
            往返总成本
        """
        buy_cost = self.calculate_buy_cost_with_code(code, amount)
        sell_cost = self.calculate_sell_cost_with_code(code, amount)
        return buy_cost.total + sell_cost.total


def calculate_cost(
    side: TradeSide,
    amount: float,
    config: Optional[CostConfig] = None
) -> float:
    """
    便捷函数：计算交易成本（不区分市场）

    注意：此函数对深圳股票会高估成本（包含过户费）。
    建议使用 calculate_cost_with_code 函数。

    Args:
        side: 交易方向
        amount: 交易金额
        config: 成本配置

    Returns:
        总成本
    """
    calculator = TransactionCostCalculator(config)
    breakdown = calculator.calculate(side, Decimal(str(amount)))
    return float(breakdown.total)


def calculate_cost_with_code(
    side: TradeSide,
    code: str,
    amount: float,
    config: Optional[CostConfig] = None
) -> float:
    """
    便捷函数：根据股票代码计算交易成本

    Args:
        side: 交易方向
        code: 股票代码
        amount: 交易金额
        config: 成本配置

    Returns:
        总成本
    """
    calculator = TransactionCostCalculator(config)
    breakdown = calculator.calculate_with_code(side, code, Decimal(str(amount)))
    return float(breakdown.total)

