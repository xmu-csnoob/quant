#!/usr/bin/env python
"""
导入股票日线数据到数据库

使用AKShare免费接口导入数据
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd

from src.data.storage.sqlite_storage import SQLiteStorage


def import_with_akshare(codes: list, start_date: str, end_date: str):
    """使用AKShare导入数据"""
    try:
        import akshare as ak
    except ImportError:
        logger.error("请先安装akshare: pip install akshare")
        return False

    storage = SQLiteStorage()
    success_count = 0

    for code in codes:
        try:
            # 分离代码和交易所
            if '.' in code:
                symbol, exchange = code.split('.')
            else:
                symbol = code
                exchange = 'SH' if code.startswith('6') else 'SZ'

            # AKShare使用纯代码
            logger.info(f"获取 {code} 数据...")

            # 使用AKShare的stock_zh_a_hist接口
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )

            if df is None or len(df) == 0:
                logger.warning(f"{code} 无数据")
                continue

            # 转换列名
            column_map = {
                '日期': 'trade_date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '换手率': 'turnover',
            }
            df = df.rename(columns=column_map)

            # 添加ts_code
            df['ts_code'] = code

            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # 保存到数据库
            storage.save_daily_prices(df, code)
            logger.success(f"{code} 导入成功: {len(df)}条")
            success_count += 1

        except Exception as e:
            logger.error(f"{code} 导入失败: {e}")
            continue

    logger.info(f"导入完成: {success_count}/{len(codes)} 只股票")
    return success_count > 0


def import_with_tushare(codes: list, start_date: str, end_date: str, token: str):
    """使用Tushare导入数据"""
    try:
        import tushare as ts
    except ImportError:
        logger.error("请先安装tushare: pip install tushare")
        return False

    ts.set_token(token)
    pro = ts.pro_api()
    storage = SQLiteStorage()
    success_count = 0

    for code in codes:
        try:
            logger.info(f"获取 {code} 数据...")

            df = pro.daily(
                ts_code=code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
            )

            if df is None or len(df) == 0:
                logger.warning(f"{code} 无数据")
                continue

            # 转换格式
            df = df.rename(columns={
                'trade_date': 'trade_date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
            })

            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # 保存到数据库
            storage.save_daily_prices(df, code)
            logger.success(f"{code} 导入成功: {len(df)}条")
            success_count += 1

        except Exception as e:
            logger.error(f"{code} 导入失败: {e}")
            continue

    logger.info(f"导入完成: {success_count}/{len(codes)} 只股票")
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="导入股票日线数据")
    parser.add_argument('--source', type=str, default='akshare', choices=['akshare', 'tushare'],
                        help='数据源 (default: akshare)')
    parser.add_argument('--codes', type=str, default=None,
                        help='股票代码列表，逗号分隔 (default: 自动选择)')
    parser.add_argument('--start', type=str, default='20230101',
                        help='开始日期 (default: 20230101)')
    parser.add_argument('--end', type=str, default=None,
                        help='结束日期 (default: 今天)')
    parser.add_argument('--token', type=str, default=None,
                        help='Tushare Token (使用Tushare时必填)')

    args = parser.parse_args()

    # 默认股票列表
    if args.codes:
        codes = [c.strip() for c in args.codes.split(',')]
    else:
        codes = [
            "600000.SH",  # 浦发银行
            "600519.SH",  # 贵州茅台
            "000858.SZ",  # 五粮液
            "601318.SH",  # 中国平安
            "000333.SZ",  # 美的集团
            "600036.SH",  # 招商银行
            "601166.SH",  # 兴业银行
            "000001.SZ",  # 平安银行
        ]

    # 日期处理
    start_date = args.start
    end_date = args.end or datetime.now().strftime('%Y%m%d')

    logger.info(f"数据源: {args.source}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info(f"股票数量: {len(codes)}")

    if args.source == 'akshare':
        success = import_with_akshare(codes, start_date, end_date)
    else:
        if not args.token:
            logger.error("使用Tushare需要提供--token参数")
            return 1
        success = import_with_tushare(codes, start_date, end_date, args.token)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
