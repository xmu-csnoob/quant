"""
数据划分工具

正确做法：
- 训练集：用于开发策略逻辑
- 验证集：用于调整参数
- 测试集：用于最终验证（只能用一次！）

错误做法：
- 用全部数据回测 → 调参 → 再回测 → 直到盈利
- 这是在"对着答案做题"，会导致过拟合
"""

from dataclasses import dataclass
from typing import Tuple
import pandas as pd


@dataclass
class DataSplit:
    """数据划分结果"""

    train: pd.DataFrame  # 训练集
    val: pd.DataFrame  # 验证集
    test: pd.DataFrame  # 测试集


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    date_col: str = "trade_date",
) -> DataSplit:
    """
    按时间顺序划分数据

    Args:
        df: 原始数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        date_col: 日期列名

    Returns:
        DataSplit 对象
    """
    # 按日期排序
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print("=" * 70)
    print("数据划分结果")
    print("=" * 70)
    print(f"总数据量: {n} 条")
    print(f"训练集: {len(train)} 条 ({train.iloc[0][date_col]} ~ {train.iloc[-1][date_col]})")
    print(f"验证集: {len(val)} 条 ({val.iloc[0][date_col]} ~ {val.iloc[-1][date_col]})")
    print(f"测试集: {len(test)} 条 ({test.iloc[0][date_col]} ~ {test.iloc[-1][date_col]})")
    print("=" * 70)

    return DataSplit(train=train, val=val, test=test)


def split_data_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
    date_col: str = "trade_date",
) -> DataSplit:
    """
    按日期划分数据

    Args:
        df: 原始数据
        train_end_date: 训练集结束日期 (YYYY-MM-DD)
        val_end_date: 验证集结束日期 (YYYY-MM-DD)
        date_col: 日期列名

    Returns:
        DataSplit 对象
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    train = df[df[date_col] < train_end_date].copy()
    val = df[(df[date_col] >= train_end_date) & (df[date_col] < val_end_date)].copy()
    test = df[df[date_col] >= val_end_date].copy()

    print("=" * 70)
    print("数据划分结果（按日期）")
    print("=" * 70)
    print(f"总数据量: {len(df)} 条")
    print(f"训练集: {len(train)} 条")
    if len(train) > 0:
        print(f"  日期范围: {train.iloc[0][date_col]} ~ {train.iloc[-1][date_col]}")
    print(f"验证集: {len(val)} 条")
    if len(val) > 0:
        print(f"  日期范围: {val.iloc[0][date_col]} ~ {val.iloc[-1][date_col]}")
    print(f"测试集: {len(test)} 条")
    if len(test) > 0:
        print(f"  日期范围: {test.iloc[0][date_col]} ~ {test.iloc[-1][date_col]}")
    print("=" * 70)
    print("\n重要提醒：")
    print("1. 训练集：用于开发策略逻辑")
    print("2. 验证集：用于调整参数（可以多次使用）")
    print("3. 测试集：用于最终验证（只能用一次！）")
    print("=" * 70)

    return DataSplit(train=train, val=val, test=test)
