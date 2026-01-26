"""
数据流完整性测试

测试完整的数据流程：
1. Fetcher 获取数据
2. Normalizer 标准化
3. Database 存储
4. Query 按需查询

验证各层之间的数据一致性
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 path
# test_data_flow.py -> data/tests -> data -> quant
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from data.fetchers.mock import MockDataFetcher
# from data.fetchers.tushare import TushareDataFetcher  # 需要 tushare
from data.fetchers.base import Exchange
from data.processors.normalizer import normalize_from_source
from data.database import Database


def test_mock_data_flow():
    """测试 Mock 数据完整流程"""
    logger.info("=" * 50)
    logger.info("测试 Mock 数据流")
    logger.info("=" * 50)

    # 1. 初始化
    fetcher = MockDataFetcher(scenario="normal")
    normalizer_source = "mock"
    db = Database(db_path="data/test/test.db")

    # 2. 初始化数据库
    db.init_db()
    logger.info("✓ 数据库初始化完成")

    # 3. 获取数据
    df_raw = fetcher.get_daily_price("600000.SH", "20230101", "20230131")
    logger.info(f"✓ 获取原始数据: {len(df_raw)} 行")
    logger.info(f"  字段: {df_raw.columns.tolist()}")

    # 4. 标准化
    df_norm = normalize_from_source(df_raw, source="mock", data_type="daily")
    logger.info(f"✓ 标准化完成: {len(df_norm)} 行")
    logger.info(f"  字段: {df_norm.columns.tolist()}")

    # 5. 存储到数据库
    rows = db.insert_dataframe(df_norm, "stock_daily", if_exists="replace")
    logger.info(f"✓ 存储到数据库: {rows} 行")

    # 6. 使用不同视图查询

    # Minimal view
    df_minimal = db.query_by_view_name(
        "minimal",
        filters={"ts_code": "600000.SH"},
        order_by="trade_date ASC"
    )
    logger.info(f"✓ Minimal 视图查询: {len(df_minimal)} 行, {len(df_minimal.columns)} 列")
    logger.info(f"  字段: {df_minimal.columns.tolist()}")

    # Standard view
    df_standard = db.query_by_view_name(
        "standard",
        filters={"ts_code": "600000.SH"},
        order_by="trade_date ASC"
    )
    logger.info(f"✓ Standard 视图查询: {len(df_standard)} 行, {len(df_standard.columns)} 列")
    logger.info(f"  字段: {df_standard.columns.tolist()}")

    # 7. 验证数据一致性
    assert len(df_minimal) == len(df_norm), "Minimal view 行数不匹配"
    assert len(df_standard) == len(df_norm), "Standard view 行数不匹配"

    logger.info("\n✓ Mock 数据流测试通过！")


def test_field_mapping():
    """测试字段映射"""
    logger.info("\n" + "=" * 50)
    logger.info("测试字段映射")
    logger.info("=" * 50)

    # 模拟 Tushare 格式（用 vol 而不是 volume）
    df_tushare_format = pd.DataFrame({
        "ts_code": ["600000.SH"] * 3,
        "trade_date": ["20230101", "20230102", "20230103"],
        "open": [10.0, 10.5, 11.0],
        "high": [10.5, 11.0, 11.5],
        "low": [9.5, 10.0, 10.5],
        "close": [10.5, 11.0, 11.5],
        "vol": [100000, 150000, 200000],  # Tushare 用 vol
        "amount": [10500, 16500, 23000],
    })

    logger.info(f"原始 Tushare 格式字段: {df_tushare_format.columns.tolist()}")

    # 标准化
    df_norm = normalize_from_source(df_tushare_format, source="tushare", data_type="daily")

    logger.info(f"标准化后字段: {df_norm.columns.tolist()}")

    # 验证 volume 字段存在
    assert "volume" in df_norm.columns, "标准化后应包含 volume 字段"
    assert "vol" not in df_norm.columns, "标准化后不应包含 vol 字段"

    logger.info("✓ 字段映射测试通过！")


def test_data_views():
    """测试不同数据视图"""
    logger.info("\n" + "=" * 50)
    logger.info("测试数据视图")
    logger.info("=" * 50)

    db = Database(db_path="data/test/test.db")

    # 获取所有可用视图
    from data.database import PREDEFINED_VIEWS

    logger.info(f"可用视图: {list(PREDEFINED_VIEWS.keys())}")

    for view_name, view in PREDEFINED_VIEWS.items():
        logger.info(f"\n视图: {view_name}")
        logger.info(f"  表: {view.table}")
        logger.info(f"  字段数: {len(view.columns)}")
        logger.info(f"  字段: {view.columns}")

        # 查询
        df = db.query_by_view_name(
            view_name,
            filters={"ts_code": "600000.SH"},
            limit=5
        )

        logger.info(f"  查询结果: {len(df)} 行")
        assert len(df.columns) == len(view.columns), f"{view_name} 视图列数不匹配"

    logger.info("\n✓ 数据视图测试通过！")


def test_stock_list_flow():
    """测试股票列表数据流"""
    logger.info("\n" + "=" * 50)
    logger.info("测试股票列表数据流")
    logger.info("=" * 50)

    fetcher = MockDataFetcher()
    db = Database(db_path="data/test/test.db")

    # 1. 获取股票列表
    df_raw = fetcher.get_stock_list(Exchange.SSE)
    logger.info(f"✓ 获取股票列表: {len(df_raw)} 只")

    # 2. 标准化
    df_norm = normalize_from_source(df_raw, source="mock", data_type="stock_list")
    logger.info(f"✓ 标准化完成")
    logger.info(f"  字段: {df_norm.columns.tolist()}")

    # 3. 存储
    rows = db.insert_dataframe(df_norm, "stock_list", if_exists="replace")
    logger.info(f"✓ 存储到数据库: {rows} 行")

    # 4. 查询
    df_query = db.query(
        "stock_list",
        filters={"exchange": "SSE"},
        order_by="ts_code ASC"
    )
    logger.info(f"✓ 查询结果: {len(df_query)} 只")

    logger.info("\n✓ 股票列表数据流测试通过！")


if __name__ == "__main__":
    # 运行所有测试
    test_mock_data_flow()
    test_field_mapping()
    test_data_views()
    test_stock_list_flow()

    logger.info("\n" + "=" * 50)
    logger.info("所有测试通过！")
    logger.info("=" * 50)
