# 数据处理模块设计文档

## 概述

数据处理模块负责将不同数据源获取的原始数据进行标准化、清洗和持久化存储，为上层应用提供统一的数据访问接口。

## 设计目标

1. **统一数据格式**：消除不同数据源（Mock/Tushare/AkShare）的格式差异
2. **数据持久化**：使用 SQLite 数据库存储全量数据
3. **按需查询**：通过视图机制实现字段按需读取
4. **符合数据库范式**：遵循第三范式（3NF）设计表结构

## 架构设计

```
┌────────────────────────────────────────────────────────────────┐
│                        数据获取层                               │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │   Mock   │  │ Tushare  │  │ AkShare  │                    │
│  │ Fetcher  │  │ Fetcher  │  │ Fetcher  │                    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                    │
│       │             │             │                           │
│       └──────────┬──┴─────────────┘                           │
│                  │ 原始数据（格式各异）                         │
└──────────────────┼────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                       标准化层                                 │
│                                                               │
│  DataNormalizer:                                             │
│  • 字段名映射 (vol → volume)                                  │
│  • 数据类型统一 (日期 → YYYYMMDD)                              │
│  • 单位统一 (amount → 千元)                                    │
│  • 补充缺失字段 (填充 NaN)                                     │
│  • 解析交易所信息 (从 ts_code)                                 │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────────┐
│                     持久化层 (SQLite)                          │
│                                                               │
│  表结构（符合 3NF）：                                          │
│  ┌─────────────┬────────────────────────────────────┐         │
│  │ stock_list  │ ts_code(PK), symbol, name, ...     │         │
│  ├─────────────┼────────────────────────────────────┤         │
│  │ stock_daily │ ts_code, trade_date(PK), open, ... │         │
│  ├─────────────┼────────────────────────────────────┤         │
│  │ index_daily │ index_code, trade_date(PK), ...    │         │
│  └─────────────┴────────────────────────────────────┘         │
└──────────────────┬────────────────────────────────────────────┘
                   │
                   ▼ SQL SELECT columns_needed
┌────────────────────────────────────────────────────────────────┐
│                       查询层                                   │
│                                                               │
│  预定义视图：                                                   │
│  ┌──────────┬──────────────────────────────────────┐          │
│  │ minimal  │ OHLCV + 基础字段 (8 列)               │          │
│  │ standard │ OHLCV + pre_close/change/pct_chg      │          │
│  │ backtest │ 回测所需字段                          │          │
│  │ indicator│ 技术指标计算字段                      │          │
│  └──────────┴──────────────────────────────────────┘          │
│                                                               │
│  自定义视图：                                                   │
│  - 通过 columns 参数指定任意字段组合                           │
└────────────────────────────────────────────────────────────────┘
```

## 模块组成

### 1. 数据库 Schema (data/database/schema.py)

**职责**：定义数据库表结构和视图

**核心类**：
- `ColumnDef`: 列定义
- `DataView`: 数据视图定义

**表结构**：

| 表名 | 主键 | 字段数 | 说明 |
|------|------|--------|------|
| stock_list | ts_code | 8 | 股票基本信息 |
| stock_daily | ts_code + trade_date | 17 | 日线行情数据 |
| index_daily | index_code + trade_date | 11 | 指数日线数据 |
| trading_calendar | exchange + cal_date | 3 | 交易日历 |

**索引设计**：
```sql
-- stock_daily
CREATE INDEX idx_stock_daily_ts_date ON stock_daily(ts_code, trade_date);
CREATE INDEX idx_stock_daily_date ON stock_daily(trade_date);
```

### 2. 数据库操作 (data/database/db.py)

**职责**：提供数据库 CRUD 操作

**核心类**：`Database`

**主要方法**：

```python
# 初始化
db = Database(db_path="data/quant.db")
db.init_db()  # 创建表和索引

# 插入数据
db.insert_dataframe(df, "stock_daily", if_exists="append")

# 查询数据 - 方式 1：通用查询
df = db.query(
    table_name="stock_daily",
    columns=["ts_code", "close"],
    filters={"ts_code": "600000.SH", "trade_date >=": "20230101"}
)

# 查询数据 - 方式 2：按视图查询
df = db.query_by_view_name(
    "minimal",
    filters={"ts_code": "600000.SH"}
)
```

### 3. 数据标准化 (data/processors/normalizer.py)

**职责**：将不同数据源的数据统一到标准格式

**核心类**：`DataNormalizer`

**标准化规则**：

| 数据源 | 字段映射 | 说明 |
|--------|----------|------|
| Tushare | vol → volume | Tushare 用 vol，标准用 volume |
| AkShare | 无映射 | AkShare 已在内部处理 |
| Mock | 无映射 | Mock 数据符合标准格式 |

**标准化流程**：

```
原始 DataFrame
       │
       ▼
1. 字段名映射
       │
       ▼
2. 验证必需字段 (ts_code, trade_date, close)
       │
       ▼
3. 日期格式统一 (YYYYMMDD)
       │
       ▼
4. 数值类型转换
       │
       ▼
5. 单位统一
       │
       ▼
6. 补充缺失字段 (填充 NaN)
       │
       ▼
7. 添加交易所信息
       │
       ▼
标准化 DataFrame
```

**使用示例**：

```python
from data.processors.normalizer import normalize_from_source

# 标准化日线数据
df_norm = normalize_from_source(
    df_raw,
    source="tushare",  # or "akshare", "mock"
    data_type="daily"
)

# 标准化股票列表
df_norm = normalize_from_source(
    df_raw,
    source="mock",
    data_type="stock_list"
)
```

## 数据视图设计

### 预定义视图

```python
# Minimal - 最小字段集
VIEW_MINIMAL = DataView(
    columns=["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount"]
)

# Standard - 标准字段集
VIEW_STANDARD = DataView(
    columns=["ts_code", "trade_date", "open", "high", "low", "close",
             "pre_close", "change", "pct_chg", "volume", "amount"]
)

# Backtest - 回测专用
VIEW_BACKTEST = DataView(
    columns=["ts_code", "trade_date", "open", "high", "low", "close",
             "pre_close", "volume", "amount", "turnover_rate"]
)

# Indicator - 技术指标计算
VIEW_INDICATOR = DataView(
    columns=["ts_code", "trade_date", "open", "high", "low", "close", "volume"]
)
```

### 自定义视图

```python
from data.database import DataView, Database

db = Database()

# 定义自定义视图
my_view = DataView(
    name="my_view",
    table="stock_daily",
    columns=["ts_code", "trade_date", "close", "volume", "amount"]
)

# 按自定义视图查询
df = db.query_by_view(my_view, filters={"ts_code": "600000.SH"})
```

## 数据流示例

```python
from data.fetchers.tushare import TushareDataFetcher
from data.processors.normalizer import normalize_from_source
from data.database import Database

# 1. 初始化
fetcher = TushareDataFetcher()
db = Database()
db.init_db()

# 2. 获取原始数据
df_raw = fetcher.get_daily_price("600000.SH", "20230101", "20231231")

# 3. 标准化
df_norm = normalize_from_source(df_raw, source="tushare", data_type="daily")

# 4. 存储到数据库
db.insert_dataframe(df_norm, "stock_daily")

# 5. 按需查询
# 回测引擎只需要部分字段
backtest_data = db.query_by_view_name(
    "backtest",
    filters={"ts_code": "600000.SH"}
)

# 技术指标计算只需要 OHLCV
indicator_data = db.query_by_view_name(
    "indicator",
    filters={"ts_code": "600000.SH"}
)
```

## 设计优势

### 1. 单一数据源
- 全量数据只存一份
- 避免数据冗余
- 保证数据一致性

### 2. 按需查询
- 通过 SQL SELECT 指定需要的字段
- 减少内存占用
- 提高查询效率

### 3. 关注点分离
- 数据获取层：只负责获取
- 标准化层：只负责格式转换
- 持久化层：只负责存储
- 查询层：只负责按需读取

### 4. 易于扩展
- 新增数据源：只需实现标准化映射
- 新增字段：修改 Schema 即可
- 新增视图：定义新的 DataView

## 目录结构

```
data/
├── database/
│   ├── __init__.py
│   ├── schema.py          # 表结构和视图定义
│   └── db.py              # 数据库操作
├── processors/
│   ├── __init__.py
│   └── normalizer.py      # 数据标准化
├── tests/
│   └── test_data_flow.py  # 数据流测试
└── design/
    └── processor.md       # 本文档
```

## 测试

运行测试验证数据流完整性：

```bash
python3 data/tests/test_data_flow.py
```

测试覆盖：
- Mock 数据完整流程
- 字段映射 (vol → volume)
- 数据视图查询
- 股票列表数据流
