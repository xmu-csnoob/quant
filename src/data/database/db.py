"""
SQLite 数据库操作层

提供：
- 数据库初始化（建表、索引）
- 数据插入/更新
- 数据查询
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Literal
from loguru import logger

from src.data.database.schema import (
    get_all_create_sql,
    get_all_indexes,
    PREDEFINED_VIEWS,
    DataView
)


class Database:
    """SQLite 数据库操作类"""

    def __init__(self, db_path: str = "data/quant.db"):
        """初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None

        logger.info(f"Database initialized: {self.db_path}")

    def connect(self) -> sqlite3.Connection:
        """获取数据库连接

        Returns:
            sqlite3.Connection
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # 允许多线程使用
            )
            # 使用 DataFrame 的行列名
            self._conn.row_factory = sqlite3.Row
            logger.debug(f"Connected to database: {self.db_path}")

        return self._conn

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def init_db(self):
        """初始化数据库（创建表和索引）"""
        conn = self.connect()
        cursor = conn.cursor()

        # 创建表
        for sql in get_all_create_sql():
            cursor.execute(sql)

        # 创建索引
        for sql in get_all_indexes():
            try:
                cursor.execute(sql)
            except sqlite3.OperationalError as e:
                logger.warning(f"Index creation warning: {e}")

        conn.commit()
        logger.info("Database initialized with tables and indexes")

    # ========================================
    # 插入操作
    # ========================================

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: Literal["fail", "replace", "append"] = "append",
        chunksize: int = 1000
    ) -> int:
        """插入 DataFrame 到数据库

        Args:
            df: 要插入的数据
            table_name: 表名
            if_exists: 表已存在时的处理方式
                - "fail": 抛出错误
                - "replace": 删除原表，重新创建
                - "append": 追加数据（默认）
            chunksize: 分批插入大小

        Returns:
            插入的行数
        """
        if df.empty:
            logger.warning(f"DataFrame is empty, skipping insert to {table_name}")
            return 0

        conn = self.connect()

        # 如果是 replace 模式，先删除表
        if if_exists == "replace":
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
            # 重新创建表
            from src.data.database.schema import ALL_TABLES
            if table_name in ALL_TABLES:
                from src.data.database.schema import get_create_table_sql
                columns, _ = ALL_TABLES[table_name]
                sql = get_create_table_sql(table_name, columns)
                cursor.execute(sql)
                conn.commit()
                logger.info(f"Replaced table: {table_name}")

        # 插入数据
        try:
            df.to_sql(
                table_name,
                conn,
                if_exists="append",  # replace 模式已处理，这里始终用 append
                index=False,
                chunksize=chunksize
            )
            rows = len(df)
            logger.debug(f"Inserted {rows} rows to {table_name}")
            return rows

        except Exception as e:
            logger.error(f"Failed to insert to {table_name}: {e}")
            raise

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        conflict_columns: List[str]
    ) -> int:
        """插入或更新 DataFrame（Upsert）

        如果主键冲突则更新，否则插入

        Args:
            df: 要插入的数据
            table_name: 表名
            conflict_columns: 冲突检测列（通常是主键）

        Returns:
            影响的行数
        """
        if df.empty:
            return 0

        conn = self.connect()
        cursor = conn.cursor()

        # 获取列名
        columns = df.columns.tolist()
        placeholders = ", ".join(["?" for _ in columns])
        col_str = ", ".join(columns)

        # 构建 INSERT OR REPLACE 语句
        sql = f"""
        INSERT OR REPLACE INTO {table_name} ({col_str})
        VALUES ({placeholders})
        """

        # 转换数据为列表
        data = df.values.tolist()

        try:
            cursor.executemany(sql, data)
            conn.commit()
            rows = cursor.rowcount
            logger.debug(f"Upserted {rows} rows to {table_name}")
            return rows

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to upsert to {table_name}: {e}")
            raise

    # ========================================
    # 查询操作
    # ========================================

    def query(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """查询数据

        Args:
            table_name: 表名
            columns: 需要的列（None 表示所有列）
            filters: 过滤条件，如 {"ts_code": "600000.SH", "trade_date >": "20230101"}
            order_by: 排序，如 "trade_date ASC"
            limit: 限制返回行数

        Returns:
            查询结果 DataFrame
        """
        conn = self.connect()

        # 构建列
        if columns:
            col_str = ", ".join(columns)
        else:
            col_str = "*"

        # 构建 SQL
        sql = f"SELECT {col_str} FROM {table_name}"

        # 构建过滤条件
        where_parts = []
        params = []

        if filters:
            for key, value in filters.items():
                # 处理操作符
                if " " in key:
                    col, op = key.split(" ", 1)
                else:
                    col, op = key, "="

                where_parts.append(f"{col} {op} ?")
                params.append(value)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        # 排序
        if order_by:
            sql += f" ORDER BY {order_by}"

        # 限制
        if limit:
            sql += f" LIMIT {limit}"

        # 执行查询
        try:
            df = pd.read_sql_query(sql, conn, params=params)
            logger.debug(f"Queried {len(df)} rows from {table_name}")
            return df

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def query_by_view(
        self,
        view: DataView,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """按视图查询

        Args:
            view: 数据视图定义
            filters: 过滤条件
            order_by: 排序
            limit: 限制行数

        Returns:
            查询结果 DataFrame
        """
        return self.query(
            table_name=view.table,
            columns=view.columns,
            filters=filters,
            order_by=order_by,
            limit=limit
        )

    def query_by_view_name(
        self,
        view_name: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """按预定义视图名称查询

        Args:
            view_name: 视图名称（minimal/standard/backtest/indicator）
            filters: 过滤条件
            order_by: 排序
            limit: 限制行数

        Returns:
            查询结果 DataFrame
        """
        if view_name not in PREDEFINED_VIEWS:
            raise ValueError(f"Unknown view: {view_name}. Available: {list(PREDEFINED_VIEWS.keys())}")

        view = PREDEFINED_VIEWS[view_name]
        return self.query_by_view(view, filters, order_by, limit)

    # ========================================
    # 便捷方法
    # ========================================

    def get_stock_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        view_name: str = "standard"
    ) -> pd.DataFrame:
        """获取股票日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            view_name: 视图名称

        Returns:
            日线数据 DataFrame
        """
        return self.query_by_view_name(
            view_name=view_name,
            filters={
                "ts_code": ts_code,
                "trade_date >=": start_date,
                "trade_date <=": end_date
            },
            order_by="trade_date ASC"
        )

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在

        Args:
            table_name: 表名

        Returns:
            是否存在
        """
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        return cursor.fetchone() is not None

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """获取表结构信息

        Args:
            table_name: 表名

        Returns:
            表结构信息 DataFrame
        """
        conn = self.connect()
        return pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
