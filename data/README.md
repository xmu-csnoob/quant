# 数据模块使用指南

## 概述

数据模块提供了完整的数据获取、存储和缓存功能，支持 Mock 数据和真实 Tushare 数据。

## 目录结构

```
data/
├── fetchers/           # 数据获取器
│   ├── base.py         # 基础类和异常定义
│   ├── mock.py         # Mock 数据获取器（模拟数据）
│   └── tushare.py      # Tushare 数据获取器（真实数据）
├── storage/            # 数据存储
│   └── storage.py      # DataStorage 类
├── cache/              # 缓存
│   └── cache.py        # DataCache 类
├── api/                # API 层
│   └── data_manager.py # DataManager 门面类
├── tests/              # 测试
│   ├── test_fetchers.py
│   └── quick_test.py
└── examples/           # 使用示例
    └── usage_example.py
```

## 快速开始

### 1. 使用 Mock 数据（开发阶段）

```python
from data.fetchers.mock import MockDataFetcher
from data.storage.storage import DataStorage
from data.api.data_manager import DataManager
from data.fetchers.base import Exchange

# 创建 Mock 数据获取器
fetcher = MockDataFetcher(scenario="normal")
storage = DataStorage(base_path="data/raw")
manager = DataManager(fetcher=fetcher, storage=storage)

# 获取股票列表
stock_list = manager.get_stock_list(Exchange.SSE)
print(stock_list)

# 获取日线数据
df = manager.get_daily_price("600000.SH", "20230101", "20231231")
print(df.head())

# 批量下载
manager.fetch_and_store(Exchange.SSE, "20230101", "20231231")
```

### 2. 使用真实 Tushare 数据

```python
import os
from data.fetchers.tushare import TushareDataFetcher

# 设置 Token
# export TUSHARE_TOKEN=your_token_here

# 创建 Tushare 数据获取器
fetcher = TushareDataFetcher()
storage = DataStorage(base_path="data/raw")
manager = DataManager(fetcher=fetcher, storage=storage)

# 获取真实数据
df = manager.get_daily_price("600000.SH", "20230101", "20231231")
```

## 支持的市场场景

MockDataFetcher 支持多种市场场景：

| 场景 | 说明 | 收益率特征 |
|------|------|-----------|
| `normal` | 正常市场 | 随机游走 |
| `bull` | 牛市 | 持续上涨（+1000%） |
| `bear` | 熊市 | 持续下跌（-90%） |
| `sideways` | 横盘 | 震荡（±5%） |
| `volatile` | 高波动 | 大幅波动 |

```python
fetcher = MockDataFetcher()

# 切换到牛市
fetcher.set_scenario("bull")
df = manager.get_daily_price("600000.SH", "20230101", "20231231")

# 切换到熊市
fetcher.set_scenario("bear")
df = manager.get_daily_price("600000.SH", "20230101", "20231231")
```

## 数据访问优先级

DataManager 按以下优先级获取数据：

1. **内存缓存** → 最快
2. **本地文件** → 次快
3. **API 获取** → 较慢（仅 Mock/Tushare）

## 运行测试

```bash
# 快速测试
python3 data/tests/quick_test.py

# 完整测试（需要 pytest）
pytest data/tests/test_fetchers.py

# 使用示例
python3 data/examples/usage_example.py
```

## 获取 Tushare Token

1. 访问 https://tushare.pro/register
2. 注册账号
3. 获取 Token
4. 设置环境变量：

```bash
export TUSHARE_TOKEN=your_token_here
```

## 配置

### 缓存配置

```python
manager = DataManager(
    fetcher=fetcher,
    storage=storage,
    cache_size=100  # 缓存 100 只股票（约 10MB）
)
```

### 存储配置

```python
storage = DataStorage(
    base_path="data/raw",        # 原始数据路径
    processed_path="data/processed",  # 处理后数据路径
    metadata_path="data/metadata",    # 元数据路径
    encoding="utf-8"
)
```

## API 文档

### DataManager

| 方法 | 说明 |
|------|------|
| `get_daily_price(ts_code, start, end)` | 获取日线数据 |
| `get_stock_list(exchange)` | 获取股票列表 |
| `fetch_and_store(exchange, start, end)` | 批量下载 |
| `update_latest(exchange, days)` | 增量更新 |
| `get_cache_stats()` | 获取缓存统计 |
| `clear_cache()` | 清空缓存 |

### DataCache

| 方法 | 说明 |
|------|------|
| `get(key)` | 获取缓存 |
| `put(key, value)` | 存入缓存 |
| `exists(key)` | 检查键是否存在 |
| `clear()` | 清空缓存 |
| `get_stats()` | 获取统计信息 |

### DataStorage

| 方法 | 说明 |
|------|------|
| `save_daily_price(df, ts_code, exchange)` | 保存日线数据 |
| `load_daily_price(ts_code, exchange)` | 加载日线数据 |
| `exists(ts_code, exchange)` | 检查文件是否存在 |
| `save_stock_list(df, exchange)` | 保存股票列表 |
| `load_stock_list(exchange)` | 加载股票列表 |

## 常见问题

### Q: 如何切换数据源？

A: 只需替换 fetcher 实例：

```python
# Mock 数据
manager.fetcher = MockDataFetcher()

# Tushare 数据
manager.fetcher = TushareDataFetcher()
```

### Q: 如何提高数据获取速度？

A:
1. 使用缓存（默认开启）
2. 批量下载而非逐只下载
3. 使用本地文件（已下载的数据）

### Q: Mock 数据和真实数据的区别？

A:
- Mock 数据：模拟生成，快速、可控、免费
- 真实数据：从 Tushare 获取，真实、准确、需要 Token

## 后续开发

- [ ] 实现 DataProcessor（数据清洗）
- [ ] 实现 DataView（数据视图层）
- [ ] 添加复权处理
- [ ] 添加技术指标计算
- [ ] 添加 AkShare 数据源

## 联系

- 项目地址：`/home/wangwenfei/quant`
- 设计文档：`data/design/`
- 测试代码：`data/tests/`
