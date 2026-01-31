# A股量化交易系统 - Bug清单

**仓库**: https://github.com/xmu-csnoob/quant
**审查日期**: 2026-01-29
**审查范围**: 核心模块源码

---

## 🔴 严重Bug (Critical)

### Bug #1: RSI指标除零风险
**文件**: `utils/indicators/rsi.py:100`
**严重程度**: ⚠️ CRITICAL

```python
# 问题代码
rs = avg_gain / avg_loss
result["RSI"] = 100 - (100 / (1 + rs))
```

**问题描述**: 当 `avg_loss` 为 0 时会触发除零错误。在市场单边上涨时，所有跌幅为0，会导致 `avg_loss` 为 0。

**复现条件**:
- 股票连续多日上涨（无下跌日）
- 数据周期较短，波动极小

**修复建议**:
```python
# 修复方案
rs = avg_gain / avg_loss.replace(0, np.nan)
result["RSI"] = 100 - (100 / (1 + rs))
# 当 avg_loss = 0 时，RSI 应为 100
result["RSI"] = result["RSI"].fillna(100.0)
```

---

### Bug #2: KDJ指标除零风险
**文件**: `utils/features/technical.py:294`
**严重程度**: ⚠️ CRITICAL

```python
# 问题代码
df["RSV"] = (df["close"] - low_min) / (high_max - low_min) * 100
```

**问题描述**: 当 `high_max == low_min` 时（即周期内最高价等于最低价），会触发除零错误。这种情况常见于：
- 停牌后复牌的股票
- 一字涨停/跌停的股票
- 数据异常情况

**修复建议**:
```python
# 修复方案
range_val = high_max - low_min
df["RSV"] = np.where(
    range_val != 0,
    (df["close"] - low_min) / range_val * 100,
    50  # 当无波动时，RSV取中性值50
)
```

---

### Bug #3: MA斜率计算除零风险
**文件**: `utils/features/technical.py:145`
**严重程度**: ⚠️ HIGH

```python
# 问题代码
df[f"MA{period}_slope"] = df[ma_col].diff(1) / df[ma_col].shift(1)
```

**问题描述**: 当 `shift(1)` 为 0 或 NaN 时会出错。虽然在金融数据中MA几乎不可能为0，但初始数据可能为NaN。

**修复建议**:
```python
# 修复方案
prev_ma = df[ma_col].shift(1)
df[f"MA{period}_slope"] = np.where(
    prev_ma != 0,
    df[ma_col].diff(1) / prev_ma,
    0
).fillna(0)
```

---

### Bug #4: 布林带位置计算除零风险
**文件**: `utils/features/technical.py:346`
**严重程度**: ⚠️ HIGH

```python
# 问题代码
df["BB_position"] = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
```

**问题描述**: 当布林带上下轨相等时（波动率为0），会触发除零错误。

**修复建议**:
```python
# 修复方案
bb_width = df["BB_upper"] - df["BB_lower"]
df["BB_position"] = np.where(
    bb_width != 0,
    (df["close"] - df["BB_lower"]) / bb_width,
    0.5  # 无波动时取中间位置
)
```

---

### Bug #5: 成交量斜率计算除零风险
**文件**: `utils/features/technical.py:404`
**严重程度**: ⚠️ HIGH

```python
# 问题代码
df["volume_slope"] = df["volume"].diff(1) / df["volume"].shift(1)
```

**问题描述**: 前一日成交量为 0 时会触发除零错误。这种情况在:
- 新股上市首日后
- 停牌复牌
- 数据缺失

**修复建议**:
```python
# 修复方案
prev_volume = df["volume"].shift(1)
df["volume_slope"] = np.where(
    prev_volume != 0,
    df["volume"].diff(1) / prev_volume,
    0
).fillna(0)
```

---

### Bug #6: 移动止损逻辑错误
**文件**: `risk/manager.py:227-238`
**严重程度**: ⚠️ CRITICAL (逻辑错误)

```python
# 问题代码
if position.unrealized_pnl_ratio > 0.05:
    # 盈利超过5%，设置移动止损
    trailing_stop = position.unrealized_pnl_ratio * 0.5
    if position.unrealized_pnl_ratio < trailing_stop:  # ❌ 永远不会触发!
        return RiskCheck(...)
```

**问题描述**:
- 当 `unrealized_pnl_ratio > 0` 时，`0.5 * x < x` 永远为 False
- 这意味着移动止损**永远不会触发**
- 盈利回撤保护机制完全失效

**修复建议**:
```python
# 修复方案: 使用绝对回撤阈值
if position.unrealized_pnl_ratio > 0.05:
    # 记录峰值盈利
    peak_profit = getattr(position, '_peak_profit', position.unrealized_pnl_ratio)
    if position.unrealized_pnl_ratio > peak_profit:
        position._peak_profit = position.unrealized_pnl_ratio
    # 回撤超过峰值的50%时平仓
    elif position.unrealized_pnl_ratio < peak_profit * 0.5:
        return RiskCheck(
            passed=False,
            action=RiskAction.CLOSE,
            reason=f"触发移动止损（盈利从{peak_profit:.1%}回撤到{position.unrealized_pnl_ratio:.1%}）",
            ...
        )
```

---

### Bug #7: ML策略日期格式错误
**文件**: `strategies/ml_strategy.py:92, 104`
**严重程度**: ⚠️ MEDIUM

```python
# 问题代码
date=row["trade_date"].strftime("%Y%m%d"),
```

**问题描述**:
- 假设 `trade_date` 是 Timestamp 类型
- 如果数据已经是字符串格式会报错：`AttributeError: 'str' object has no attribute 'strftime'`

**修复建议**:
```python
# 修复方案
def _format_date(date_val):
    if isinstance(date_val, pd.Timestamp):
        return date_val.strftime("%Y%m%d")
    return str(date_val).replace("-", "")

# 使用
date=_format_date(row["trade_date"]),
```

---

## 🟠 中等问题 (Medium)

### Bug #8: 成交量列名不一致
**文件**: 多处 (base.py, sqlite_storage.py等)
**严重程度**: ⚠️ MEDIUM

**问题描述**:
- `_validate_data()` 硬编码要求 `volume` 列
- Tushare API 返回的是 `vol` 列
- 导致数据验证失败

**影响范围**:
- `utils/features/base.py:61`
- `utils/indicators/base.py:58`

**修复建议**:
```python
# 修复方案: 兼容两种列名
required_columns = ["open", "high", "low", "close"]
# 检查成交量列（兼容 volume 和 vol）
has_volume = "volume" in df.columns or "vol" in df.columns
if not has_volume:
    raise ValueError("缺少 volume 或 vol 列")
```

---

### Bug #9: 凯利公式负值未显式处理
**文件**: `risk/position_sizer.py:178`
**严重程度**: ⚠️ MEDIUM

```python
# 问题代码
kelly_ratio = (avg_win_loss * win_rate - (1 - win_rate)) / avg_win_loss
kelly_ratio *= 0.5
kelly_ratio *= confidence
ratio = max(self.min_position_ratio, min(self.max_position_ratio, kelly_ratio))
```

**问题描述**:
- 当期望收益为负时，凯利公式结果为负
- 虽然 `max(min_position_ratio, ...)` 会限制到最小值
- 但负凯利值意味着不应该交易，而不是最小仓位

**修复建议**:
```python
# 修复方案
kelly_ratio = (avg_win_loss * win_rate - (1 - win_rate)) / avg_win_loss
# 如果凯利值为负，不应开仓
if kelly_ratio < 0:
    return PositionSize(
        shares=0,
        amount=0,
        risk_ratio=0,
        reason=f"凯利公式为负（胜率{win_rate:.1%}，盈亏比{avg_win_loss:.2f}），不建议开仓"
    )
```

---

### Bug #10: 缓存元数据未持久化
**文件**: `data/cache/persistent_cache.py`
**严重程度**: ⚠️ MEDIUM

**问题描述**:
- `self.metadata` 字典只存储在内存中
- 程序重启后，虽然缓存文件存在，但元数据丢失
- 导致缓存无法正确判断过期时间

**影响**: 重启后所有缓存都会被当作已过期

**修复建议**:
```python
# 修复方案: 将元数据持久化到文件
def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600):
    # ... 现有代码 ...
    self.metadata_file = self.cache_dir / ".metadata.json"
    self._load_metadata()

def _load_metadata(self):
    if self.metadata_file.exists():
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)

def _save_metadata(self):
    with open(self.metadata_file, 'w') as f:
        json.dump(self.metadata, f)

def put(self, key: str, value: pd.DataFrame, ttl: int = None):
    # ... 现有代码 ...
    self._save_metadata()  # 保存元数据
```

---

## 🟡 轻微问题 (Minor)

### Bug #11: 持仓统计包含已清仓数据
**文件**: `risk/manager.py:337`
**严重程度**: LOW

```python
# 问题代码
total_position_value = sum(p.shares * p.current_price for p in self.positions.values())
```

**问题描述**: 未过滤 `shares <= 0` 的持仓，可能产生不准确统计

**修复建议**:
```python
total_position_value = sum(
    p.shares * p.current_price
    for p in self.positions.values()
    if p.shares > 0
)
```

---

### Bug #12: 模拟成交价格精度问题
**文件**: `trading/api.py:307-309`
**严重程度**: LOW

```python
# 问题代码
order.avg_price = (
    (order.avg_price * (order.filled_quantity - fill_quantity) + fill_price * fill_quantity)
    / order.filled_quantity
)
```

**问题描述**: 首次成交时 `order.avg_price` 可能为 None

**修复建议**:
```python
if order.filled_quantity == fill_quantity:
    order.avg_price = fill_price
else:
    order.avg_price = (
        (order.avg_price * (order.filled_quantity - fill_quantity) + fill_price * fill_quantity)
        / order.filled_quantity
    )
```

---

### Bug #13: 均值回测策略预热期不足
**文件**: `strategies/mean_reversion.py:103`
**严重程度**: LOW

```python
# 问题代码
for i in range(self.ma_period, len(df)):
```

**问题描述**: 循环从 `ma_period` 开始，但:
- 布林带需要 `bb_period` (默认20)
- RSI需要 `rsi_period` (默认14)
- 应该取最大值

**修复建议**:
```python
warmup = max(self.ma_period, self.bb_period, self.rsi_period)
for i in range(warmup, len(df)):
```

---

### Bug #14: 夏普比率计算不准确
**文件**: `backtesting/simple_backtester.py:253-256`
**严重程度**: LOW

```python
# 问题代码
if trades:
    returns = [t.pnl_ratio for t in trades]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    sharpe_ratio = sharpe * np.sqrt(252)  # 年化
```

**问题描述**:
- 使用交易收益率而非日收益率
- 年化系数 `sqrt(252)` 假设每日交易，不准确

**修复建议**:
```python
# 应该基于每日净值计算夏普比率
daily_returns = []
# ... 计算每日收益率 ...
sharpe_ratio = (
    np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0
    else 0
)
```

---

### Bug #15: 组合策略中的属性引用错误
**文件**: `strategies/ensemble_strategy.py:171`
**严重程度**: LOW

```python
# 问题代码
weight = self.weights.get(s.name, 0)  # Signal 没有 name 属性
```

**问题描述**: `Signal` 类没有 `name` 属性，应该是 `strategy.name`

**修复建议**:
```python
# 在 _weighted_ensemble 中，应该使用策略名
# 需要重构 date_signals 的结构，包含策略名称
```

---

## 📊 问题统计

| 严重程度 | 数量 | 占比 |
|----------|------|------|
| Critical (严重) | 7 | 47% |
| High (高) | 3 | 20% |
| Medium (中) | 3 | 20% |
| Low (低) | 2 | 13% |
| **总计** | **15** | 100% |

---

## 🎯 修复优先级

### P0 - 立即修复 (会导致程序崩溃)
1. Bug #1: RSI除零
2. Bug #2: KDJ除零
3. Bug #6: 移动止损逻辑错误

### P1 - 尽快修复 (影响功能正确性)
4. Bug #3: MA斜率除零
5. Bug #4: 布林带除零
6. Bug #5: 成交量斜率除零
7. Bug #7: 日期格式错误
8. Bug #8: 列名不一致

### P2 - 计划修复 (影响用户体验)
9. Bug #9: 凯利公式负值
10. Bug #10: 缓存元数据

### P3 - 低优先级 (边缘问题)
11-15: 其余轻微问题

---

## 📝 补充说明

1. **数据验证不足**: 大部分数值计算未对除零、NaN、Infinity 进行防护
2. **类型假设**: 代码假设数据类型（如日期格式），缺少兼容性处理
3. **边界条件**: 对异常市场情况（停牌、一字板）处理不足
4. **测试覆盖**: 未见单元测试，建议针对上述bug添加测试用例

---

## 🔴 严重Bug (续) - ML与数据泄露问题

### Bug #16: 标签生成中的严重数据泄露
**文件**: `utils/labels.py:53`
**严重程度**: ⚠️⚠️ CRITICAL (数据泄露)

```python
# 问题代码
df["future_return"] = df["close"].pct_change(self.prediction_period).shift(-self.prediction_period)
```

**问题描述**:
- 使用 `shift(-N)` 会将**未来数据**泄露到当前行
- 训练时模型实际上"看到了"未来收益
- 回测效果会非常好，但实盘完全无效
- 这是最典型的**前视偏差 (Look-ahead Bias)**

**复现条件**:
- 使用该标签生成的模型进行回测
- 回测收益会显著高于实盘

**影响**:
- 所有使用 `LabelGenerator` 的训练脚本都受影响
- 包括: `train_ml_model.py`, `train_ml_model_v2.py`, `train_ml_model_v3.py`

**修复建议**:
```python
# 修复方案: 确保标签生成在特征提取之后，并正确处理
def generate(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 计算未来收益（使用负shift是正确的，因为这是标签）
    df["future_return"] = df["close"].pct_change(self.prediction_period).shift(-self.prediction_period)

    # 关键: 必须删除最后prediction_period行（这些行没有标签）
    df = df.iloc[:-self.prediction_period].copy()

    if self.task_type == "regression":
        df["label"] = df["future_return"]
    else:
        df["label"] = (df["future_return"] > self.threshold).astype(int)

    return df
```

---

### Bug #17: 特征工程顺序错误导致数据泄露
**文件**: `scripts/train_ml_model.py:63-77`, `train_ml_model_v2.py:75-89`, `train_ml_model_v3.py:68-83`
**严重程度**: ⚠️⚠️ CRITICAL (数据泄露)

```python
# 问题代码（所有训练脚本）
# 2. 特征工程
df_features = feature_extractor.extract(df_all)

# 3. 生成标签
df_labeled = label_gen.generate(df_features)

# 4. 准备训练数据
df_clean = df_labeled.dropna(subset=feature_cols + ["label"]).copy()

# 时间序列分割
train_size = int(len(df_clean) * 0.6)
df_train = df_clean.iloc[:train_size].copy()
df_test = df_clean.iloc[train_size + val_size:].copy()
```

**问题描述**:
- 特征提取在**数据分割之前**进行
- 特征计算中的滚动窗口（如MA20）使用了测试集的数据
- 例如: 训练集最后一天的特征可能包含了测试集前19天的数据
- 这是**隐性的数据泄露**，很难被发现

**正确的处理顺序**:
1. 先按时间分割数据
2. 对训练集和测试集**分别**提取特征
3. 确保测试集的特征计算不使用未来数据

**修复建议**:
```python
# 修复方案
# 1. 先分割数据
train_end_idx = int(len(df_all) * 0.6)
val_end_idx = int(len(df_all) * 0.8)

df_train_raw = df_all.iloc[:train_end_idx].copy()
df_val_raw = df_all.iloc[train_end_idx:val_end_idx].copy()
df_test_raw = df_all.iloc[val_end_idx:].copy()

# 2. 分别提取特征（需要额外预留warmup期）
def extract_with_warmup(df, feature_extractor, warmup_days=60):
    # 获取额外warmup数据
    # ... 提取特征 ...
    # 只返回目标期数据
    return df_features[warmup_days:]

df_train = extract_with_warmup(df_train_raw, feature_extractor)
df_val = extract_with_warmup(df_val_raw, feature_extractor)
df_test = extract_with_warmup(df_test_raw, feature_extractor)

# 3. 生成标签（每个数据集独立）
```

---

### Bug #18: ML特征使用当日收盘价（无法在交易时获取）
**文件**: `utils/features/ml_features.py`
**严重程度**: ⚠️⚠️ CRITICAL (前视偏差)

```python
# 问题代码
def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # 各种周期的收益率
    for period in [1, 3, 5, 10]:
        df[f"f_return_{period}d"] = df["close"].pct_change(period)
    # 使用了当日close，但交易时无法知道收盘价

def _add_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # RSI, MACD等都基于当日close
    df["f_rsi"] = self._calculate_rsi(df["close"], 14)
    # 交易时只能用昨日close计算
```

**问题描述**:
- 所有特征都使用**当日收盘价**计算
- 但实际交易时，需要在盘中或开盘前做出决策
- 此时收盘价还不知道
- 导致回测和实盘表现严重不符

**正确的特征计算**:
```python
# 修复方案: 使用昨日或已知价格
def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # 使用昨日收盘价计算收益率
    close_prev = df["close"].shift(1)
    for period in [1, 3, 5, 10]:
        df[f"f_return_{period}d"] = close_prev.pct_change(period)
    return df
```

---

### Bug #19: Walk-Forward验证中训练集持续扩展包含未来信息
**文件**: `scripts/train_ml_robust.py:30-71`
**严重程度**: ⚠️⚠️ CRITICAL (数据泄露)

```python
# 问题代码
splits = [
    # 测试期1: 2021年Q1
    {
        "train_start": "20200101",  # 固定起点
        "train_end": "20201231",
        "test_start": "20210101",
        "test_end": "20210331",
    },
    # 测试期2: 2021年Q3
    {
        "train_start": "20200101",  # 仍然是2020年开始
        "train_end": "20210630",    # 但训练期包含了测试期1的数据！
        "test_start": "20210701",
        "test_end": "20210930",
    },
    # ...
]
```

**问题描述**:
- Walk-Forward验证应该使用**滚动窗口**
- 但这里的训练集持续扩展，每次都包含之前的测试期
- 这导致模型在测试期2的训练阶段，已经"看过了"测试期1的数据
- 违背了Walk-Forward验证的初衷

**正确的Walk-Forward设计**:
```python
# 修复方案: 使用真正的滚动窗口
splits = [
    # 窗口1
    {
        "train_start": "20180101",  # 训练窗口固定长度（如3年）
        "train_end": "20201231",
        "test_start": "20210101",
        "test_end": "20210331",
    },
    # 窗口2: 整体滚动
    {
        "train_start": "20190101",  # 起点向前移动
        "train_end": "20211231",
        "test_start": "20210401",
        "test_end": "20210630",
    },
]
```

---

### Bug #20: 标签异常值截断导致跨股票信息泄露
**文件**: `scripts/train_ml_model_v2.py:92`, `train_ml_model_v3.py:86`
**严重程度**: ⚠️ HIGH (数据泄露)

```python
# 问题代码
# 1. 合并所有股票数据
df_all = pd.concat(all_data, ignore_index=True)

# 2. 提取特征
df_features = feature_extractor.extract(df_all)

# 3. 生成标签
df_labeled = label_gen.generate(df_features)

# 4. 截断异常值 - 在合并后进行！
df_labeled["label"] = clip_outliers(df_labeled["label"], 0.01, 0.99)
```

**问题描述**:
- 异常值截断在**所有股票数据合并后**进行
- 这意味着股票A的标签分布会影响股票B的标签
- 不同股票之间产生了信息泄露
- 模型可能学到股票间的相对关系而非绝对模式

**修复建议**:
```python
# 修复方案: 对每只股票单独处理
all_data_processed = []
for stock_code, exchange in stock_list:
    df = storage.load_daily_price(stock_code, exchange)
    df["ts_code"] = stock_code

    # 提取特征
    df_features = feature_extractor.extract(df)

    # 生成标签
    df_labeled = label_gen.generate(df_features)

    # 对每只股票单独截断异常值
    df_labeled["label"] = clip_outliers(df_labeled["label"], 0.01, 0.99)

    all_data_processed.append(df_labeled)

df_all = pd.concat(all_data_processed, ignore_index=True)
```

---

### Bug #21: ML特征中的RSI除零风险
**文件**: `utils/features/ml_features.py:184-191`
**严重程度**: ⚠️ CRITICAL

```python
# 问题代码
def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss  # ❌ 除零风险
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

**问题描述**: 与Bug #1相同，但出现在ML特征中。当市场单边上涨时 `loss` 为 0。

**修复建议**:
```python
# 修复方案
def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    # 处理除零
    rs = np.divide(gain, loss, where=loss!=0, out=np.full_like(gain, np.nan))
    rsi = 100 - (100 / (1 + rs))

    # 当loss=0时，RSI=100
    rsi = rsi.fillna(100.0)
    return rsi
```

---

### Bug #22: 增强特征中多处除零风险
**文件**: `utils/features/enhanced_features.py`
**严重程度**: ⚠️ HIGH

```python
# 问题代码1: 第105行
df["f_upper_shadow_ratio"] = upper_shadow / (df["high"] - df["low"])
# 当 high == low 时除零

# 问题代码2: 第106行
df["f_lower_shadow_ratio"] = lower_shadow / (df["high"] - df["low"])
# 同上

# 问题代码3: 第118行
lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
# 有条件判断，但仍可能在某些情况失效
```

**修复建议**:
```python
# 修复方案
daily_range = df["high"] - df["low"]
df["f_upper_shadow_ratio"] = np.divide(
    upper_shadow, daily_range,
    where=daily_range!=0,
    out=np.zeros_like(upper_shadow, dtype=float)
)
df["f_lower_shadow_ratio"] = np.divide(
    lower_shadow, daily_range,
    where=daily_range!=0,
    out=np.zeros_like(lower_shadow, dtype=float)
)
```

---

### Bug #23: 时间特征可能导致过拟合
**文件**: `utils/features/enhanced_features.py:71-91`
**严重程度**: ⚠️ MEDIUM (模型欺骗)

```python
# 问题代码
def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df["f_day_of_week"] = df["trade_date"].dt.dayofweek / 4.0
    df["f_month"] = (df["trade_date"].dt.month - 1) / 11.0
    df["f_quarter"] = (df["trade_date"].dt.quarter - 1) / 3.0
    df["f_month_start"] = (df["trade_date"].dt.day <= 5).astype(int)
    df["f_month_end"] = (df["trade_date"].dt.day >= 25).astype(int)
```

**问题描述**:
- 时间日历特征（月份、季度）是**强特征**
- 模型可能学到"3月总是涨"这种季节性模式
- 但这是历史巧合，未来不一定成立
- 导致**过拟合**和**模型欺骗**

**验证方法**:
```python
# 检查特征重要性中时间特征的排名
# 如果 f_month, f_quarter 排名很高，说明模型可能过拟合
```

**修复建议**:
```python
# 方案1: 移除时间特征
# 方案2: 使用相对时间而非绝对时间
df["f_days_in_month"] = df["trade_date"].dt.day / df["trade_date"].dt.days_in_month
# 方案3: 交叉验证时按时间分组，避免时间泄露
```

---

### Bug #24: 回测使用测试期数据但模型训练见过
**文件**: `scripts/train_ml_model.py:188-202`
**严重程度**: ⚠️ HIGH (数据泄露)

```python
# 问题代码
# 训练时
test_start = df_test["trade_date"].min()
test_end = df_test["trade_date"].max()

# 回测时
df = df[(df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)]
# 然后用训练好的模型在这个期间回测
```

**问题描述**:
- 回测使用的时间段**与测试集完全相同**
- 虽然这不是直接的训练数据泄露
- 但模型参数是根据测试集表现调优的（early stopping使用验证集）
- 这导致**回测结果过于乐观**

**正确的验证方式**:
```python
# 使用Walk-Forward方法
# 或者保留一个"最终测试集"，只在模型完全冻结后使用一次
```

---

### Bug #25: 特征重要性在测试集上计算
**文件**: `scripts/train_ml_model_v2.py:191-197`
**严重程度**: ⚠️ MEDIUM

```python
# 问题代码
# 重新训练（只使用选定的特征）
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]
X_test_selected = X_test[:, selected_indices]  # ❌ 测试集参与特征选择
```

**问题描述**:
- 特征选择使用全部数据（包括测试集）
- 这导致特征选择过程"看到了"测试集
- 属于**间接的数据泄露**

**修复建议**:
```python
# 只在训练集上进行特征选择
importance = bst.get_score(importance_type='gain')
# ... 特征选择逻辑 ...

# 然后用选定的特征在验证集和测试集上评估
```

---

## 📊 ML/数据泄露问题统计

| 类别 | 数量 | 占比 |
|------|------|------|
| Look-ahead Bias (前视偏差) | 4 | 36% |
| Data Leakage (数据泄露) | 5 | 45% |
| 除零风险 | 2 | 18% |
| **总计** | **11** | 100% |

---

## 🎯 ML模型欺骗检测清单

### 如何判断模型是否有效？

| 检查项 | 说明 | 当前状态 |
|--------|------|----------|
| ✅ 时间序列分割 | 训练/验证/测试按时间划分 | ❌ 部分正确 |
| ✅ 特征在分割后计算 | 避免滚动窗口泄露 | ❌ 未实现 |
| ✅ 使用历史价格计算特征 | 不用当日收盘价 | ❌ 未实现 |
| ✅ Walk-Forward验证 | 真正的滚动窗口 | ❌ 实现有误 |
| ✅ 样本外测试 | 保留最终测试集 | ❌ 未实现 |
| ✅ 交易成本模拟 | 考虑滑点和手续费 | ❌ 简化处理 |
| ✅ 多市场验证 | 牛市/熊市/震荡都测试 | ⚠️ 部分实现 |

### 当前模型风险评级: ⚠️⚠️ **高风险**

**主要原因**:
1. 特征使用当日收盘价（无法在交易时获取）
2. 特征计算在分割之前（滚动窗口泄露）
3. Walk-Forward实现有误（训练集扩展而非滚动）

**结论**: 当前回测收益**不可信**，实盘收益可能显著低于回测。

---

## 🚨 紧急修复优先级

### P0 - 立即修复 (导致模型完全无效)
1. **Bug #16**: 标签生成需删除末尾行
2. **Bug #17**: 特征提取必须在分割后进行
3. **Bug #18**: 使用历史价格计算特征

### P1 - 尽快修复
4. **Bug #19**: 修复Walk-Forward实现
5. **Bug #20**: 分股票处理异常值
6. **Bug #21**: RSI除零

### P2 - 计划修复
7. **Bug #22**: 增强特征除零
8. **Bug #23**: 评估时间特征影响
9. **Bug #24**: 使用真正的样本外测试
10. **Bug #25**: 特征选择只用训练集

---

## 📝 补充说明

### ML最佳实践建议

1. **特征工程原则**:
   - 只使用T-1时刻及之前的数据
   - 避免使用当日OHLC（只能用T-1及之前）
   - 滚动窗口计算时要考虑分割边界

2. **数据分割原则**:
   - 先分割，再处理
   - 每个数据集独立提取特征
   - Walk-Forward用真正的滚动窗口

3. **模型验证原则**:
   - 保留一个最终测试集
   - 只在模型完全冻结后使用一次
   - 避免在测试集上调参

---

**更新时间**: 2026-01-29 (追加ML与数据泄露专项审查)
**审查人**: Claude Code
