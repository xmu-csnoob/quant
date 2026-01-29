"""
SQLite数据存储

将数据存储到SQLite数据库，提高查询性能和便利性
"""

import sqlite3
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pandas as pd
from loguru import logger

from data.fetchers.base import Exchange


class SQLiteStorage:
    """
    SQLite数据存储

    特点：
    1. 单文件存储所有数据
    2. 支持SQL查询
    3. 索引加速查询
    4. 事务保证数据一致性
    """

    def __init__(self, db_path: str = "data/quant.db"):
        """
        初始化SQLite存储

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30
        )

        # 启用外键约束
        self.conn.execute("PRAGMA foreign_keys = ON")
        # 设置WAL模式提高并发性能
        self.conn.execute("PRAGMA journal_mode = WAL")

        self._create_tables()

        logger.info(f"SQLiteStorage initialized: {self.db_path}")

    def _create_tables(self):
        """创建数据表"""

        # 股票日线数据表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                pre_close REAL,
                change REAL,
                pct_chg REAL,
                vol REAL,
                amount REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
        """)

        # 创建索引
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_date
            ON daily_prices(trade_date)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_code
            ON daily_prices(ts_code)
        """)

        # 股票列表表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_list (
                ts_code TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                area TEXT,
                industry TEXT,
                market TEXT,
                list_date TEXT,
                updated_at TEXT
            )
        """)

        # 交易记录表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                amount REAL NOT NULL,
                reason TEXT
            )
        """)

        # 持仓记录表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                current_price REAL,
                market_value REAL,
                unrealized_pnl REAL,
                updated_at TEXT
            )
        """)

        self.conn.commit()

    def save_daily_prices(self, df: pd.DataFrame, ts_code: str) -> int:
        """
        保存日线数据

        Args:
            df: 日线数据DataFrame
            ts_code: 股票代码

        Returns:
            保存的行数
        """
        if df.empty:
            return 0

        # 添加股票代码列
        df = df.copy()
        df['ts_code'] = ts_code

        # 重命名列以匹配数据库
        column_map = {
            'vol': 'vol',  # Tushare使用vol
            'volume': 'vol',  # 兼容volume列名
        }

        # 保存到数据库（使用INSERT OR IGNORE处理重复）
        try:
            # 使用executemany批量插入
            cursor = self.conn.executemany(
                """INSERT OR IGNORE INTO daily_prices
                   (ts_code, trade_date, open, high, low, close, pre_close,
                    change, pct_chg, vol, amount)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'pre_close', 'change', 'pct_chg', 'vol', 'amount']].values.tolist()
            )
            self.conn.commit()
            logger.debug(f"Saved {cursor.rowcount} rows for {ts_code}")
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to save {ts_code}: {e}")
            self.conn.rollback()
            return 0

    def get_daily_prices(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)

        Returns:
            DataFrame
        """
        query = "SELECT * FROM daily_prices WHERE ts_code = ?"
        params = [ts_code]

        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)

        query += " ORDER BY trade_date"

        df = pd.read_sql_query(query, self.conn, params=params)

        # 重命名vol列为volume（兼容）
        if 'vol' in df.columns and 'volume' not in df.columns:
            df = df.rename(columns={'vol': 'volume'})

        return df

    def get_all_stocks(self) -> List[str]:
        """获取所有股票代码"""
        cursor = self.conn.execute(
            "SELECT DISTINCT ts_code FROM daily_prices ORDER BY ts_code"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, ts_code: str) -> tuple:
        """获取股票的数据日期范围"""
        cursor = self.conn.execute(
            """SELECT MIN(trade_date), MAX(trade_date)
               FROM daily_prices
               WHERE ts_code = ?""",
            (ts_code,)
        )
        return cursor.fetchone()

    def save_trade(
        self,
        symbol: str,
        trade_date: str,
        side: str,
        price: float,
        quantity: int,
        reason: str = ""
    ):
        """保存交易记录"""
        self.conn.execute(
            """INSERT INTO trades
               (symbol, trade_date, side, price, quantity, amount, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, trade_date, side, price, quantity, price * quantity, reason)
        )
        self.conn.commit()

    def get_trades(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """获取交易记录"""
        if symbol:
            query = "SELECT * FROM trades WHERE symbol = ? ORDER BY trade_date"
            df = pd.read_sql_query(query, self.conn, params=[symbol])
        else:
            query = "SELECT * FROM trades ORDER BY trade_date"
            df = pd.read_sql_query(query, self.conn)

        return df

    def save_position(
        self,
        symbol: str,
        quantity: int,
        avg_cost: float,
        current_price: float,
        market_value: float,
        unrealized_pnl: float
    ):
        """保存/更新持仓"""
        self.conn.execute(
            """INSERT OR REPLACE INTO positions
               (symbol, quantity, avg_cost, current_price, market_value, unrealized_pnl, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, quantity, avg_cost, current_price, market_value,
             unrealized_pnl, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_positions(self) -> pd.DataFrame:
        """获取所有持仓"""
        query = "SELECT * FROM positions WHERE quantity > 0"
        return pd.read_sql_query(query, self.conn)

    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        stats = {}

        # 股票数量
        cursor = self.conn.execute(
            "SELECT COUNT(DISTINCT ts_code) FROM daily_prices"
        )
        stats['stock_count'] = cursor.fetchone()[0]

        # 数据行数
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM daily_prices"
        )
        stats['total_rows'] = cursor.fetchone()[0]

        # 日期范围
        cursor = self.conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices"
        )
        min_date, max_date = cursor.fetchone()
        stats['date_range'] = f"{min_date} ~ {max_date}"

        # 交易记录数
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM trades"
        )
        stats['trade_count'] = cursor.fetchone()[0]

        # 数据库文件大小
        if self.db_path.exists():
            stats['db_size_mb'] = self.db_path.stat().st_size / 1024 / 1024

        return stats

    def export_to_csv(self, output_dir: str = "data/csv_export"):
        """导出数据到CSV文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 导出日线数据
        for ts_code in self.get_all_stocks():
            df = self.get_daily_prices(ts_code)
            if not df.empty:
                # 转换股票代码为文件名
                filename = ts_code.replace('.', '_') + '.csv'
                df.to_csv(output_path / filename, index=False)

        logger.info(f"Exported data to {output_dir}")

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __del__(self):
        """析构函数"""
        try:
            self.conn.close()
        except:
            pass
