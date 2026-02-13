"""
滑点模型模块

实现不同的滑点模拟策略：
- 固定滑点
- 基于成交量的滑点
- 随机滑点
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
import random
from typing import Optional

# 从 costs 模块导入共享的 TradeSide
from src.backtesting.costs import TradeSide


@dataclass
class SlippageResult:
    """
    滑点计算结果

    Attributes:
        original_price: 原始价格
        adjusted_price: 调整后价格（含滑点）
        slippage_amount: 滑点金额
        slippage_rate: 滑点比例
    """

    original_price: Decimal
    adjusted_price: Decimal
    slippage_amount: Decimal
    slippage_rate: Decimal

    def __str__(self) -> str:
        return (
            f"Slippage({self.original_price:.3f} -> {self.adjusted_price:.3f}, "
            f"slippage={self.slippage_rate*100:.4f}%)"
        )


class BaseSlippageModel(ABC):
    """
    滑点模型基类

    所有滑点模型都应继承此类并实现 apply_slippage 方法
    """

    @abstractmethod
    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> SlippageResult:
        """
        应用滑点

        Args:
            price: 原始价格
            side: 交易方向
            volume: 交易量（可选，用于成交量滑点模型）
            avg_volume: 平均成交量（可选，用于成交量滑点模型）

        Returns:
            滑点计算结果
        """
        pass

    def get_slippage_multiplier(self, side: TradeSide, slippage_rate: Decimal) -> Decimal:
        """
        获取滑点调整系数

        买入时价格上调，卖出时价格下调

        Args:
            side: 交易方向
            slippage_rate: 滑点比例

        Returns:
            价格调整系数
        """
        if side == TradeSide.BUY:
            # 买入时，实际成交价更高
            return Decimal("1") + slippage_rate
        else:
            # 卖出时，实际成交价更低
            return Decimal("1") - slippage_rate


class NoSlippage(BaseSlippageModel):
    """
    无滑点模型

    用于对比测试或理想情况假设
    """

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> SlippageResult:
        """无滑点"""
        return SlippageResult(
            original_price=price,
            adjusted_price=price,
            slippage_amount=Decimal("0"),
            slippage_rate=Decimal("0"),
        )


class FixedSlippage(BaseSlippageModel):
    """
    固定滑点模型

    使用固定的滑点比例

    Example:
        >>> model = FixedSlippage(slippage_rate=Decimal('0.001'))  # 0.1%
        >>> result = model.apply_slippage(Decimal('10'), TradeSide.BUY)
        >>> print(result.adjusted_price)  # 10.01
    """

    def __init__(self, slippage_rate: Decimal = Decimal("0.001")):
        """
        初始化固定滑点模型

        Args:
            slippage_rate: 滑点比例（默认0.1%）
        """
        self.slippage_rate = slippage_rate

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> SlippageResult:
        """应用固定滑点"""
        multiplier = self.get_slippage_multiplier(side, self.slippage_rate)
        adjusted_price = price * multiplier
        slippage_amount = abs(adjusted_price - price)

        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_rate=self.slippage_rate,
        )


class VolumeBasedSlippage(BaseSlippageModel):
    """
    基于成交量的滑点模型

    滑点与成交量占比成正比：
    - 成交量占比越高，滑点越大
    - 适合大单交易模拟

    Example:
        >>> model = VolumeBasedSlippage(base_rate=Decimal('0.001'), max_rate=Decimal('0.01'))
        >>> result = model.apply_slippage(
        ...     Decimal('10'), TradeSide.BUY,
        ...     volume=Decimal('10000'), avg_volume=Decimal('50000')
        ... )
    """

    def __init__(
        self,
        base_rate: Decimal = Decimal("0.0005"),
        max_rate: Decimal = Decimal("0.02"),
        volume_impact_factor: Decimal = Decimal("0.5"),
    ):
        """
        初始化成交量滑点模型

        Args:
            base_rate: 基础滑点率（默认0.05%）
            max_rate: 最大滑点率（默认2%）
            volume_impact_factor: 成交量影响系数（默认0.5）
        """
        self.base_rate = base_rate
        self.max_rate = max_rate
        self.volume_impact_factor = volume_impact_factor

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> SlippageResult:
        """应用基于成交量的滑点"""
        # 如果没有成交量信息，使用基础滑点
        if volume is None or avg_volume is None or avg_volume == 0:
            return FixedSlippage(self.base_rate).apply_slippage(price, side)

        # 计算成交量占比
        volume_ratio = volume / avg_volume

        # 滑点 = 基础滑点 + 成交量占比影响
        # 使用幂函数让小单滑点较小，大单滑点增长更快
        volume_impact = (volume_ratio ** self.volume_impact_factor) * self.base_rate
        slippage_rate = min(self.base_rate + volume_impact, self.max_rate)

        multiplier = self.get_slippage_multiplier(side, slippage_rate)
        adjusted_price = price * multiplier
        slippage_amount = abs(adjusted_price - price)

        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_rate=slippage_rate,
        )


class RandomSlippage(BaseSlippageModel):
    """
    随机滑点模型

    在给定范围内随机生成滑点，模拟真实市场的不确定性

    Example:
        >>> model = RandomSlippage(min_rate=Decimal('0.0005'), max_rate=Decimal('0.002'))
        >>> result = model.apply_slippage(Decimal('10'), TradeSide.BUY)
    """

    def __init__(
        self,
        min_rate: Decimal = Decimal("0.0002"),
        max_rate: Decimal = Decimal("0.002"),
        seed: Optional[int] = None,
    ):
        """
        初始化随机滑点模型

        Args:
            min_rate: 最小滑点率（默认0.02%）
            max_rate: 最大滑点率（默认0.2%）
            seed: 随机种子（用于可重复测试）
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> SlippageResult:
        """应用随机滑点"""
        # 在范围内随机生成滑点率
        slippage_rate = Decimal(str(
            random.uniform(float(self.min_rate), float(self.max_rate))
        ))

        multiplier = self.get_slippage_multiplier(side, slippage_rate)
        adjusted_price = price * multiplier
        slippage_amount = abs(adjusted_price - price)

        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_rate=slippage_rate,
        )


class MarketImpactSlippage(BaseSlippageModel):
    """
    市场冲击滑点模型

    综合考虑成交量和市场深度的滑点模型
    适用于大单交易和流动性较差的股票

    公式: slippage = base_rate + alpha * (volume / avg_volume)^beta

    其中:
    - base_rate: 基础滑点
    - alpha: 冲击系数
    - beta: 弹性系数（通常为0.5-1.0）
    """

    def __init__(
        self,
        base_rate: Decimal = Decimal("0.0005"),
        alpha: Decimal = Decimal("0.005"),
        beta: Decimal = Decimal("0.5"),
        max_rate: Decimal = Decimal("0.05"),
    ):
        """
        初始化市场冲击滑点模型

        Args:
            base_rate: 基础滑点率
            alpha: 市场冲击系数
            beta: 弹性系数
            max_rate: 最大滑点率
        """
        self.base_rate = base_rate
        self.alpha = alpha
        self.beta = beta
        self.max_rate = max_rate

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> SlippageResult:
        """应用市场冲击滑点"""
        if volume is None or avg_volume is None or avg_volume == 0:
            return FixedSlippage(self.base_rate).apply_slippage(price, side)

        # 计算市场冲击
        volume_ratio = float(volume / avg_volume)
        impact = self.alpha * Decimal(str(volume_ratio ** float(self.beta)))
        slippage_rate = min(self.base_rate + impact, self.max_rate)

        multiplier = self.get_slippage_multiplier(side, slippage_rate)
        adjusted_price = price * multiplier
        slippage_amount = abs(adjusted_price - price)

        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_rate=slippage_rate,
        )


def get_default_slippage_model() -> BaseSlippageModel:
    """
    获取默认滑点模型

    Returns:
        默认的固定滑点模型（0.1%）
    """
    return FixedSlippage(slippage_rate=Decimal("0.001"))
