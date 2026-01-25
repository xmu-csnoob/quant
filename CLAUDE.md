# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an A-Share (Chinese stock market) quantitative trading system built in Python. The goal is to develop and backtest quantitative strategies for the Shanghai (SSE), Shenzhen (SZSE), and Beijing (BSE) stock exchanges.

## Learning Path (自底向上)

**For new developers, start with `tutorial/` directory:**

1. `tutorial/01-basics/` → 什么是 OHLC 数据
2. `tutorial/02-indicators/` → 如何计算技术指标（MA）
3. `tutorial/03-signals/` → 如何生成交易信号
4. `tutorial/04-backtest/` → 如何回测策略
5. `tutorial/05-architecture/` → 完整系统架构

**Design documents** in `designs/`:
- `designs/architecture/` → 系统架构图
- `designs/class/` → 类图
- `designs/sequence/` → 时序图

**Data module design** in `data/design/`:
- `data/design/overview.md` → 数据模块概述
- `data/design/classes.md` → 类设计（含完整字段定义）
- `data/design/api.md` → API 接口设计
- `data/design/data_sources.md` → 数据来源策略

## Current Implementation Status

### ✅ Completed (2026-01-25)

**1. Data Module (数据模块) - 100% 完成**
- ✅ `data/fetchers/base.py` - 基础类和异常定义
- ✅ `data/fetchers/mock.py` - Mock 数据获取器（支持 9 种市场场景）
- ✅ `data/fetchers/tushare.py` - Tushare 真实数据获取器
- ✅ `data/cache/cache.py` - LRU 缓存
- ✅ `data/storage/storage.py` - 文件存储
- ✅ `data/api/data_manager.py` - 数据管理器（门面类）
- ✅ 所有测试通过（100%）

**2. Design Documents**
- ✅ 项目结构和配置
- ✅ 系统架构设计文档
- ✅ 数据模块详细设计

**3. Tutorial**
- ✅ 自底向上学习路径

### 🚧 Development Priority

**Phase 2: 数据处理层**
- [ ] `data/processors/processor.py` - 数据清洗和验证
- [ ] `data/processors/adjust.py` - 复权处理
- [ ] `utils/indicators/` - 技术指标计算

**Phase 3: 策略模块**
- [ ] `strategies/stock_selection/` - 选股策略
- [ ] `strategies/timing/` - 择时策略

**Phase 4: 回测引擎**
- [ ] `backtesting/engines/` - 回测引擎
- [ ] `backtesting/metrics/` - 绩效指标

## Architecture

```
data/              → 数据模块（当前重点）
├── fetchers/      → 数据获取器（Mock/Tushare）
├── storage/       → 文件存储
├── cache/         → LRU 缓存
├── api/           → 数据管理器
└── tests/         → 测试

strategies/        → 策略模块（下一阶段）
backtesting/       → 回测引擎
trading/           → 订单管理
live_trading/      → 实盘交易
risk_management/   → 风险管理
analysis/          → 绩效分析
```

## Data Module Usage

### Quick Start

```python
# 使用 Mock 数据（开发阶段）
from data.fetchers.mock import MockDataFetcher
from data.storage.storage import DataStorage
from data.api.data_manager import DataManager
from data.fetchers.base import Exchange

# 初始化
fetcher = MockDataFetcher(scenario="bull")
storage = DataStorage()
manager = DataManager(fetcher=fetcher, storage=storage)

# 获取数据
df = manager.get_daily_price("600000.SH", "20230101", "20231231")

# 批量下载
manager.fetch_and_store(Exchange.SSE, "20230101", "20231231")
```

### Using Real Data (Tushare)

```python
import os
from data.fetchers.tushare import TushareDataFetcher

# 设置 Token
export TUSHARE_TOKEN=your_token_here

# 使用真实数据
fetcher = TushareDataFetcher()
manager = DataManager(fetcher=fetcher, storage=storage)

# 获取真实数据
df = manager.get_daily_price("600000.SH", "20230101", "20231231")
```

**注意**：免费账户有频率限制（约每分钟 1 次）

### Testing

```bash
# 快速测试
python3 data/tests/quick_test.py

# 使用示例
python3 data/examples/usage_example.py

# 完整演示
python3 demo.py
```

## Data Sources

### Available Data Sources

| Source | Type | Cost | Frequency Limit | Status |
|--------|------|------|-----------------|--------|
| **Mock Data** | Simulated | Free | None | ✅ Recommended for development |
| **Tushare** | Real API | Free tier available | ~1 request/min | ✅ Available (has limits) |
| **AkShare** | Web scraping | Free | None | ⏳ Not implemented |

**Recommendation**:
- **Development**: Use Mock data (fast, controllable, unlimited)
- **Validation**: Use Tushare (real data, but has frequency limits)
- **Production**: Upgrade Tushare or implement AkShare

### Mock Data Market Scenarios

```python
fetcher = MockDataFetcher(scenario="bull")   # 牛市
fetcher = MockDataFetcher(scenario="bear")   # 熊市
fetcher = MockDataFetcher(scenario="sideways")  # 横盘
fetcher = MockDataFetcher(scenario="volatile")  # 高波动
```

## Configuration

### Environment Variables

```bash
# Tushare Token (optional, for real data)
export TUSHARE_TOKEN=your_token_here
```

### Dependencies

```bash
pip install pandas loguru tenacity tushare
```

## A-Share Trading Rules

- **T+1 system**: Stocks bought today can only be sold tomorrow
- **Trading hours**: 9:30-11:30, 13:00-15:00 (Beijing time)
- **Lot size**: Minimum 100 shares (1手)
- **Price limits**: Main board ±10%, ChiNext/STAR ±20%, BSE ±30%, ST ±5%

## Commit Message Format

Follow the format:

```
<type>(<scope>): <subject>

<body>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
**Scopes**: `data`, `strategies`, `backtesting`, `trading`, `docs`

**Example**:
```
feat(data): 实现 MockDataFetcher

- 支持 9 种市场场景
- 支持完整的 OHLCV 数据生成
- 所有测试通过

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Code Conventions

- **Docstrings**: Use Chinese for docstrings and comments
- **Type hints**: Encouraged for better code clarity
- **Logging**: Use `loguru` for logging
- **API tokens**: Never commit tokens to git, use environment variables

## Project Statistics (2026-01-25)

- **Total Python files**: 30+
- **Total lines of code**: ~2,000 lines
- **Test coverage**: 100% for data module
- **Documentation**: 7 design documents + tutorials

## Next Steps

1. ✅ Data module completed
2. 🚧 Implement technical indicators (Phase 2)
3. 📋 Develop stock selection strategies (Phase 3)
4. 📋 Build backtesting engine (Phase 4)

## Resources

- **Tutorials**: `tutorial/`
- **Design Documents**: `data/design/`, `designs/`
- **Data Module Guide**: `data/README.md`
- **Quick Test**: `data/tests/quick_test.py`
