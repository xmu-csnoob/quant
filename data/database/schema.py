"""
数据库 Schema 定义

遵循第三范式 (3NF) 设计表结构：
- 每个表有明确的主键
- 消除传递依赖
- 合理使用外键

表结构：
- stock_list: 股票基本信息
- stock_daily: 日线行情数据
- index_daily: 指数日线数据
- trading_calendar: 交易日历
"""

from dataclasses import dataclass
from typing import Literal
from enum import Enum


class Exchange(Enum):
    """交易所枚举"""
    SSE = "SSE"   # 上海证券交易所
    SZSE = "SZSE" # 深圳证券交易所
    BSE = "BSE"   # 北京证券交易所


@dataclass
class ColumnDef:
    """列定义"""
    name: str
    type: Literal["TEXT", "REAL", "INTEGER", "DATETIME"]
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    default: any = None


# ========================================
# 表结构定义
# ========================================

TABLE_STOCK_LIST = [
    ColumnDef("ts_code", "TEXT", primary_key=True),
    ColumnDef("symbol", "TEXT", nullable=False),  # 股票代码（数字部分）
    ColumnDef("name", "TEXT", nullable=False),     # 股票名称
    ColumnDef("area", "TEXT"),                     # 地域
    ColumnDef("industry", "TEXT"),                 # 行业
    ColumnDef("market", "TEXT"),                   # 市场（主板/科创板/创业板/北交所）
    ColumnDef("list_date", "TEXT"),                # 上市日期 YYYYMMDD
    ColumnDef("exchange", "TEXT", nullable=False), # 交易所 (SSE/SZSE/BSE)
]

TABLE_STOCK_DAILY = [
    ColumnDef("ts_code", "TEXT", nullable=False),
    ColumnDef("trade_date", "TEXT", nullable=False),  # 交易日期 YYYYMMDD
    ColumnDef("open", "REAL"),                         # 开盘价
    ColumnDef("high", "REAL"),                         # 最高价
    ColumnDef("low", "REAL"),                          # 最低价
    ColumnDef("close", "REAL"),                        # 收盘价
    ColumnDef("pre_close", "REAL"),                    # 昨收价
    ColumnDef("change", "REAL"),                       # 涨跌额
    ColumnDef("pct_chg", "REAL"),                      # 涨跌幅 (%)
    ColumnDef("volume", "REAL"),                       # 成交量（手）
    ColumnDef("amount", "REAL"),                       # 成交额（千元）
    ColumnDef("turnover_rate", "REAL"),                # 换手率 (%)
    ColumnDef("pe_ttm", "REAL"),                       # 市盈率 TTM
    ColumnDef("pb_mrq", "REAL"),                       # 市净率
    ColumnDef("total_mv", "REAL"),                     # 总市值（万元）
    ColumnDef("circ_mv", "REAL"),                      # 流通市值（万元）
    ColumnDef("exchange", "TEXT"),                     # 交易所 (SSE/SZSE/BSE)
]

TABLE_INDEX_DAILY = [
    ColumnDef("index_code", "TEXT", nullable=False),
    ColumnDef("trade_date", "TEXT", nullable=False),
    ColumnDef("open", "REAL"),
    ColumnDef("high", "REAL"),
    ColumnDef("low", "REAL"),
    ColumnDef("close", "REAL"),
    ColumnDef("pre_close", "REAL"),
    ColumnDef("change", "REAL"),
    ColumnDef("pct_chg", "REAL"),
    ColumnDef("volume", "REAL"),
    ColumnDef("amount", "REAL"),
]

TABLE_TRADING_CALENDAR = [
    ColumnDef("exchange", "TEXT", nullable=False),
    ColumnDef("cal_date", "TEXT", nullable=False),    # 日历日期 YYYYMMDD
    ColumnDef("is_open", "INTEGER", nullable=False),  # 是否交易 (1=是, 0=否)
]


# ========================================
# 索引定义
# ========================================

INDEXES = [
    # stock_daily
    "CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_date ON stock_daily(ts_code, trade_date);",
    "CREATE INDEX IF NOT EXISTS idx_stock_daily_date ON stock_daily(trade_date);",

    # index_daily
    "CREATE INDEX IF NOT EXISTS idx_index_daily_code_date ON index_daily(index_code, trade_date);",
    "CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily(trade_date);",

    # trading_calendar
    "CREATE INDEX IF NOT EXISTS idx_calendar_exch_date ON trading_calendar(exchange, cal_date);",
]


# ========================================
# 数据视图定义（供查询层使用）
# ========================================

@dataclass
class DataView:
    """数据视图定义"""
    name: str
    table: str
    columns: list[str]  # 需要的字段


# 预定义常用视图
VIEW_MINIMAL = DataView(
    name="minimal",
    table="stock_daily",
    columns=["ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount"]
)

VIEW_STANDARD = DataView(
    name="standard",
    table="stock_daily",
    columns=["ts_code", "trade_date", "open", "high", "low", "close",
             "pre_close", "change", "pct_chg", "volume", "amount"]
)

VIEW_BACKTEST = DataView(
    name="backtest",
    table="stock_daily",
    columns=["ts_code", "trade_date", "open", "high", "low", "close",
             "pre_close", "volume", "amount", "turnover_rate"]
)

VIEW_INDICATOR = DataView(
    name="indicator",
    table="stock_daily",
    columns=["ts_code", "trade_date", "open", "high", "low", "close", "volume"]
)

# 所有预定义视图
PREDEFINED_VIEWS = {
    "minimal": VIEW_MINIMAL,
    "standard": VIEW_STANDARD,
    "backtest": VIEW_BACKTEST,
    "indicator": VIEW_INDICATOR,
}


def get_create_table_sql(table_name: str, columns: list[ColumnDef]) -> str:
    """生成建表 SQL

    Args:
        table_name: 表名
        columns: 列定义列表

    Returns:
        CREATE TABLE SQL 语句
    """
    col_defs = []
    primary_keys = []

    for col in columns:
        parts = [col.name, col.type]

        if col.primary_key:
            parts.append("PRIMARY KEY")
            primary_keys.append(col.name)
        elif not col.nullable:
            parts.append("NOT NULL")

        if col.unique and not col.primary_key:
            parts.append("UNIQUE")

        if col.default is not None:
            parts.append(f"DEFAULT {repr(col.default)}")

        col_defs.append(" ".join(parts))

    # 如果没有主键定义，创建复合主键
    if not primary_keys and table_name == "stock_daily":
        col_defs.append("PRIMARY KEY (ts_code, trade_date)")
    elif not primary_keys and table_name == "index_daily":
        col_defs.append("PRIMARY KEY (index_code, trade_date)")

    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n  " + ",\n  ".join(col_defs) + "\n);"


# 所有表的建表 SQL
ALL_TABLES = {
    "stock_list": (TABLE_STOCK_LIST, None),
    "stock_daily": (TABLE_STOCK_DAILY, None),
    "index_daily": (TABLE_INDEX_DAILY, None),
    "trading_calendar": (TABLE_TRADING_CALENDAR, "PRIMARY KEY (exchange, cal_date)"),
}


def get_all_create_sql() -> list[str]:
    """获取所有建表 SQL"""
    sqls = []
    for table_name, (columns, pk) in ALL_TABLES.items():
        sqls.append(get_create_table_sql(table_name, columns))
    return sqls


def get_all_indexes() -> list[str]:
    """获取所有索引 SQL"""
    return INDEXES
