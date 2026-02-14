"""
滑点模型模块

实现不同的滑点模拟策略：
- 固定滑点
- 基于成交量的滑点
- 随机滑点
- 时段感知滑点（开盘/收盘滑点更大）
- 市场冲击滑点

时段系数说明：
- 集合竞价 (9:15-9:25): +100% (流动性最低)
- 开盘半小时 (9:30-10:00): +50% (波动大)
- 尾盘半小时 (14:30-15:00): +30% (集中交易)
- 其他时段: 基准滑点
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from datetime import time
import random
from typing import Optional
from enum import Enum

# 从 costs 模块导入共享的 TradeSide
from src.backtesting.costs import TradeSide


class TradingSession(Enum):
    """交易时段"""
    PRE_MARKET = "pre_market"      # 盘前 (9:15-9:25 集合竞价)
    OPENING = "opening"            # 开盘 (9:30-10:00)
    MORNING = "morning"            # 上午 (10:00-11:30)
    NOON = "noon"                  # 中午休市
    AFTERNOON = "afternoon"        # 下午 (13:00-14:30)
    CLOSING = "closing"            # 尾盘 (14:30-15:00)
    AFTER_HOURS = "after_hours"    # 盘后


@dataclass
class TradingTimeContext:
    """交易时间上下文"""
    session: TradingSession
    current_time: Optional[time] = None

    @classmethod
    def from_time(cls, t: time) -> "TradingTimeContext":
        """从时间判断交易时段"""
        if t < time(9, 15):
            return cls(TradingSession.PRE_MARKET, t)
        elif time(9, 15) <= t < time(9, 30):
            return cls(TradingSession.PRE_MARKET, t)
        elif time(9, 30) <= t < time(10, 0):
            return cls(TradingSession.OPENING, t)
        elif time(10, 0) <= t < time(11, 30):
            return cls(TradingSession.MORNING, t)
        elif time(11, 30) <= t < time(13, 0):
            return cls(TradingSession.NOON, t)
        elif time(13, 0) <= t < time(14, 30):
            return cls(TradingSession.AFTERNOON, t)
        elif time(14, 30) <= t < time(15, 0):
            return cls(TradingSession.CLOSING, t)
        else:
            return cls(TradingSession.AFTER_HOURS, t)


# 时段滑点调整系数
SESSION_MULTIPLIERS = {
    TradingSession.PRE_MARKET: Decimal("2.0"),    # 集合竞价滑点翻倍
    TradingSession.OPENING: Decimal("1.5"),       # 开盘+50%
    TradingSession.MORNING: Decimal("1.0"),       # 正常
    TradingSession.NOON: Decimal("1.0"),          # 正常
    TradingSession.AFTERNOON: Decimal("1.0"),     # 正常
    TradingSession.CLOSING: Decimal("1.3"),       # 尾盘+30%
    TradingSession.AFTER_HOURS: Decimal("3.0"),   # 盘后滑点极大
}


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


class TimeAwareSlippage(BaseSlippageModel):
    """
    时段感知滑点模型

    根据交易时段调整滑点大小：
    - 集合竞价: 基准滑点 × 2.0
    - 开盘半小时: 基准滑点 × 1.5
    - 尾盘半小时: 基准滑点 × 1.3
    - 其他时段: 基准滑点

    同时考虑成交量影响和流动性系数

    Example:
        >>> from datetime import time
        >>> model = TimeAwareSlippage(base_rate=Decimal('0.001'))
        >>> # 开盘时段买入
        >>> result = model.apply_slippage(
        ...     Decimal('10'), TradeSide.BUY,
        ...     current_time=time(9, 45)
        ... )
    """

    def __init__(
        self,
        base_rate: Decimal = Decimal("0.001"),
        volume_impact_factor: Decimal = Decimal("0.3"),
        liquidity_factor: Decimal = Decimal("0.2"),
        max_rate: Decimal = Decimal("0.03"),
    ):
        """
        初始化时段感知滑点模型

        Args:
            base_rate: 基础滑点率（默认0.1%）
            volume_impact_factor: 成交量影响系数
            liquidity_factor: 流动性影响系数
            max_rate: 最大滑点率
        """
        self.base_rate = base_rate
        self.volume_impact_factor = volume_impact_factor
        self.liquidity_factor = liquidity_factor
        self.max_rate = max_rate

    def get_session_multiplier(
        self,
        current_time: Optional[time] = None,
        session: Optional[TradingSession] = None
    ) -> Decimal:
        """
        获取时段系数

        Args:
            current_time: 当前时间
            session: 交易时段（如果已知）

        Returns:
            时段系数
        """
        if session is None and current_time is not None:
            ctx = TradingTimeContext.from_time(current_time)
            session = ctx.session
        elif session is None:
            session = TradingSession.MORNING  # 默认普通时段

        return SESSION_MULTIPLIERS.get(session, Decimal("1.0"))

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
        current_time: Optional[time] = None,
        session: Optional[TradingSession] = None,
        liquidity_score: Optional[Decimal] = None,
    ) -> SlippageResult:
        """
        应用时段感知滑点

        Args:
            price: 原始价格
            side: 交易方向
            volume: 交易量
            avg_volume: 平均成交量
            current_time: 当前时间
            session: 交易时段
            liquidity_score: 流动性评分 (0-1，1表示流动性最好)

        Returns:
            滑点计算结果
        """
        # 基础滑点
        slippage_rate = self.base_rate

        # 1. 时段调整
        session_multiplier = self.get_session_multiplier(current_time, session)
        slippage_rate = slippage_rate * session_multiplier

        # 2. 成交量影响
        if volume is not None and avg_volume is not None and avg_volume > 0:
            volume_ratio = float(volume / avg_volume)
            volume_impact = Decimal(str(volume_ratio ** float(self.volume_impact_factor)))
            slippage_rate = slippage_rate + self.base_rate * volume_impact

        # 3. 流动性影响
        if liquidity_score is not None:
            # 流动性越低，滑点越大
            # liquidity_score = 1 时，滑点不变
            # liquidity_score = 0.5 时，滑点 +20%
            # liquidity_score = 0 时，滑点 +40%
            liquidity_adjustment = (Decimal("1") - liquidity_score) * self.liquidity_factor
            slippage_rate = slippage_rate * (Decimal("1") + liquidity_adjustment)

        # 限制最大滑点
        slippage_rate = min(slippage_rate, self.max_rate)

        multiplier = self.get_slippage_multiplier(side, slippage_rate)
        adjusted_price = price * multiplier
        slippage_amount = abs(adjusted_price - price)

        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_rate=slippage_rate,
        )


class ComprehensiveSlippage(BaseSlippageModel):
    """
    综合滑点模型

    结合多种因素的滑点模型：
    1. 时段因素（开盘/收盘滑点大）
    2. 成交量因素（大单滑点大）
    3. 流动性因素（低流动性滑点大）
    4. 波动率因素（高波动滑点大）

    公式:
    slippage = base_rate × session_mult × (1 + volume_impact + volatility_adj + liquidity_adj)
    """

    def __init__(
        self,
        base_rate: Decimal = Decimal("0.001"),
        max_rate: Decimal = Decimal("0.05"),
        volume_exponent: Decimal = Decimal("0.5"),
        volatility_impact: Decimal = Decimal("0.3"),
        liquidity_impact: Decimal = Decimal("0.2"),
    ):
        """
        初始化综合滑点模型

        Args:
            base_rate: 基础滑点率
            max_rate: 最大滑点率
            volume_exponent: 成交量指数
            volatility_impact: 波动率影响系数
            liquidity_impact: 流动性影响系数
        """
        self.base_rate = base_rate
        self.max_rate = max_rate
        self.volume_exponent = volume_exponent
        self.volatility_impact = volatility_impact
        self.liquidity_impact = liquidity_impact

        # 内部使用TimeAwareSlippage处理时段
        self._time_model = TimeAwareSlippage(base_rate=base_rate)

    def apply_slippage(
        self,
        price: Decimal,
        side: TradeSide,
        volume: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
        current_time: Optional[time] = None,
        session: Optional[TradingSession] = None,
        liquidity_score: Optional[Decimal] = None,
        volatility: Optional[Decimal] = None,
    ) -> SlippageResult:
        """
        应用综合滑点

        Args:
            price: 原始价格
            side: 交易方向
            volume: 交易量
            avg_volume: 平均成交量
            current_time: 当前时间
            session: 交易时段
            liquidity_score: 流动性评分 (0-1)
            volatility: 波动率 (如日内振幅)

        Returns:
            滑点计算结果
        """
        # 基础滑点
        slippage_rate = self.base_rate

        # 1. 时段调整
        session_multiplier = self._time_model.get_session_multiplier(current_time, session)
        slippage_rate = slippage_rate * session_multiplier

        # 2. 成交量影响
        volume_adjustment = Decimal("0")
        if volume is not None and avg_volume is not None and avg_volume > 0:
            volume_ratio = float(volume / avg_volume)
            volume_adjustment = Decimal(str(volume_ratio ** float(self.volume_exponent)))
            # 大单额外加成
            if volume_ratio > 0.1:  # 超过平均成交量10%
                slippage_rate = slippage_rate + self.base_rate * volume_adjustment

        # 3. 流动性影响
        if liquidity_score is not None:
            liquidity_adjustment = (Decimal("1") - liquidity_score) * self.liquidity_impact
            slippage_rate = slippage_rate * (Decimal("1") + liquidity_adjustment)

        # 4. 波动率影响
        if volatility is not None:
            # 波动率越高，滑点越大
            # 假设正常波动率为0.02 (2%)
            normal_volatility = Decimal("0.02")
            volatility_ratio = volatility / normal_volatility
            if volatility_ratio > Decimal("1"):
                volatility_adjustment = (volatility_ratio - Decimal("1")) * self.volatility_impact
                slippage_rate = slippage_rate * (Decimal("1") + volatility_adjustment)

        # 限制最大滑点
        slippage_rate = min(slippage_rate, self.max_rate)

        multiplier = self.get_slippage_multiplier(side, slippage_rate)
        adjusted_price = price * multiplier
        slippage_amount = abs(adjusted_price - price)

        return SlippageResult(
            original_price=price,
            adjusted_price=adjusted_price,
            slippage_amount=slippage_amount,
            slippage_rate=slippage_rate,
        )

