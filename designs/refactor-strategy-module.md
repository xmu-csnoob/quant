# 策略模块重构总结

> 重构日期：2026-01-26
> 重构原因：改进架构设计，解耦策略与回测

---

## 一、重构背景

### 原架构的问题

**问题1：策略分类不通用**
```
原设计：
strategies/
├── stock_selection/  # 选股策略
├── timing/           # 择时策略
└── hedging/          # 对冲策略

问题：
- 这是中国特色的分类，不是业界标准
- 国际通用的是按策略逻辑分类（趋势/反转/套利等）
```

**问题2：回测引擎位置混乱**
```
原设计：
backtesting/
├── simple_backtester.py
└── tests/
    └── test_strategy_backtest.py  # ✗ 为什么测试在这里？

问题：
- 策略测试应该在 tests/ 目录
- 回测引擎和策略混在一起
- 职责不清晰
```

---

## 二、重构内容

### 1. 改进策略分类

#### 新增枚举类型

```python
class StrategyType(Enum):
    """策略类型（按策略逻辑分类）"""
    TREND_FOLLOWING = "trend_following"  # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"    # 均值回归
    MOMENTUM = "momentum"                # 动量
    BREAKOUT = "breakout"                # 突破
    ARBITRAGE = "arbitrage"              # 套利
    STATISTICAL = "statistical"          # 统计套利
    MARKET_MAKING = "market_making"      # 做市


class AssetClass(Enum):
    """资产类别"""
    STOCK = "stock"      # 股票
    FUTURES = "futures"  # 期货
    OPTIONS = "options"  # 期权
    FOREX = "forex"      # 外汇
    CRYPTO = "crypto"    # 加密货币
    MULTI_ASSET = "multi_asset"  # 多资产


class Frequency(Enum):
    """交易频率"""
    TICK = "tick"      # 逐笔
    MINUTE_1 = "1m"    # 1分钟
    HOUR_1 = "1h"      # 1小时
    DAILY = "1d"       # 日线
    WEEKLY = "1w"      # 周线
```

#### 策略基类添加元数据

```python
class BaseStrategy(ABC):
    # 策略元数据（子类覆盖）
    strategy_type: ClassVar[StrategyType] = StrategyType.TREND_FOLLOWING
    asset_class: ClassVar[AssetClass] = AssetClass.STOCK
    frequency: ClassVar[Frequency] = Frequency.DAILY

    def get_metadata(self) -> dict:
        """获取策略元数据"""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "asset_class": self.asset_class.value,
            "frequency": self.frequency.value,
        }
```

### 2. 重构目录结构

#### 重构前

```
strategies/
├── __init__.py
├── base.py
├── ma_macd_rsi.py
├── stock_selection/    # 功能分类（不通用）
├── timing/
└── hedging/

backtesting/
├── simple_backtester.py
└── tests/             # ✗ 测试在这里？
    └── test_strategy_backtest.py
```

#### 重构后

```
strategies/              # 策略模块（只管策略逻辑）
├── __init__.py
├── base.py            # 添加了 StrategyType, AssetClass, Frequency
├── ma_macd_rsi.py     # 添加了策略元数据
├── stock_selection/
├── timing/
└── hedging/

backtesting/             # 回测引擎（独立模块）
├── simple_backtester.py
├── engines/           # 更复杂的回测引擎（待实现）
├── metrics/           # 绩效指标（待实现）
└── (tests 目录已移除)

tests/                   # 统一测试目录
├── strategies/        # 策略测试
│   ├── test_strategy_backtest.py
│   └── debug_strategy.py
├── backtesting/       # 回测引擎测试
├── unit/              # 单元测试
└── integration/       # 集成测试

experiments/             # 实验和跑批
├── strategy_results/  # 回测结果
└── README.md
```

---

## 三、重构后的优势

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| **分类方式** | 按功能（选股/择时/对冲） | 按策略逻辑（趋势/反转/套利） |
| **通用性** | 中国特色 | 国际通用 |
| **策略元数据** | 无 | 有（type/asset/frequency） |
| **测试代码** | 混在 backtesting/tests | 统一在 tests/strategies |
| **职责分离** | 策略和回测混在一起 | 策略、回测、测试分离 |
| **可扩展性** | 低 | 高 |

---

## 四、新的设计原则

### 1. 策略只负责信号生成

```python
class BaseStrategy(ABC):
    """
    策略只负责：
    - 生成交易信号
    - 计算技术指标

    不管：
    - 回测（由回测引擎负责）
    - 订单管理（由交易系统负责）
    - 风险控制（由风控模块负责）
    """
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        pass
```

### 2. 回测引擎独立于策略

```python
class BacktestEngine:
    """
    回测引擎职责：
    - 接收任何实现了 BaseStrategy 的策略
    - 模拟真实交易环境
    - 计算绩效指标

    不依赖具体策略实现
    """
    def run(self, strategy: BaseStrategy, data: pd.DataFrame):
        pass
```

### 3. 测试代码统一管理

```
tests/
├── strategies/   # 测试策略逻辑
├── backtesting/  # 测试回测引擎
├── unit/         # 单元测试
└── integration/  # 集成测试
```

---

## 五、使用示例

### 创建新策略

```python
from strategies import BaseStrategy, StrategyType, AssetClass, Frequency

class MyStrategy(BaseStrategy):
    """我的策略"""

    # 策略元数据
    strategy_type = StrategyType.MOMENTUM
    asset_class = AssetClass.STOCK
    frequency = Frequency.DAILY

    def generate_signals(self, df):
        # 实现信号生成逻辑
        pass
```

### 运行回测

```python
from strategies import MyStrategy
from backtesting import BacktestEngine

# 创建策略
strategy = MyStrategy()

# 运行回测
engine = BacktestEngine(initial_capital=100000)
result = engine.run(strategy, data)

# 查看结果
result.print_summary()
```

---

## 六、后续改进方向

### 短期

- [ ] 实现更多策略类型
  - 均值回归策略
  - 动量策略
  - 突破策略

- [ ] 完善回测引擎
  - 支持手续费
  - 支持滑点
  - 支持多股票组合

### 中期

- [ ] 实现策略参数优化
- [ ] 实现策略组合管理
- [ ] 添加更多绩效指标

### 长期

- [ ] 实现事件驱动回测引擎
- [ ] 支持实盘交易对接
- [ ] 添加机器学习策略支持

---

## 七、关键洞察

> **"好的架构不是一蹴而就的，而是通过实践不断改进的。"**

> **"重构的价值：让代码更清晰、更易维护、更易扩展。"**

> **"策略和回测解耦：策略只管信号，回测只管执行。"**

---

**重构版本**：v1.0
**重构完成时间**：2026-01-26
**下次重构计划**：根据实际使用反馈
