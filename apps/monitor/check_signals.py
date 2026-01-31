#!/usr/bin/env python3
"""
检查当前持仓的卖出信号

基于最新收盘价检查是否需要卖出
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
import xgboost as xgb


def check_position_signal(ts_code, decision_date, storage, model, feature_extractor):
    """
    检查单个股票的信号

    Args:
        ts_code: 股票代码
        decision_date: 决策日期 (YYYYMMDD)
        storage: 数据存储
        model: XGBoost模型
        feature_extractor: 特征提取器

    Returns:
        (prob, current_price) or (None, None)
    """
    try:
        # 获取历史数据（决策日之前）
        end_date = pd.to_datetime(decision_date).strftime('%Y%m%d')
        start_date = (pd.to_datetime(decision_date) - timedelta(days=120)).strftime('%Y%m%d')

        df = storage.get_daily_prices(ts_code, start_date, end_date)

        if df is None or len(df) < 60:
            return None, None

        # 提取特征
        features = feature_extractor.extract(df)

        if len(features) < 1:
            return None, None

        # 获取最后一行（决策日当天）
        latest = features.iloc[-1]
        feature_cols = [c for c in features.columns if c.startswith('f_')]

        if len(feature_cols) == 0:
            return None, None

        X = latest[feature_cols].values.reshape(1, -1)
        prob = model.predict(xgb.DMatrix(X))[0]

        current_price = latest['close']

        return prob, current_price

    except Exception as e:
        print(f"  {ts_code}: 错误 - {e}")
        return None, None


def main():
    """检查当前持仓的信号"""
    print("=" * 60)
    print("当前持仓信号检查")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 当前持仓
    positions = {
        "000536.SZ": {"entry_price": 4.08, "quantity": 73500, "entry_date": "2025-01-06"},
        "000586.SZ": {"entry_price": 8.51, "quantity": 24600, "entry_date": "2025-01-06"},
        "000548.SZ": {"entry_price": 4.88, "quantity": 30100, "entry_date": "2025-01-06"},
    }

    # 加载模型
    model = xgb.Booster()
    model.load_model('models/xgboost_2022_2026.json')
    feature_extractor = EnhancedFeatureExtractor()
    storage = SQLiteStorage()

    # 使用最新的交易日
    decision_date = '20250129'
    sell_threshold = 0.48
    buy_threshold = 0.52

    print(f"\n决策日期: {decision_date}")
    print(f"卖出阈值: 概率 < {sell_threshold}")
    print(f"买入阈值: 概率 > {buy_threshold}")
    print()

    sell_signals = []
    hold_signals = []

    for ts_code, pos in positions.items():
        prob, current_price = check_position_signal(
            ts_code, decision_date, storage, model, feature_extractor
        )

        if prob is None:
            print(f"{ts_code}: 无法获取信号")
            continue

        # 计算当前盈亏
        cost_basis = pos['quantity'] * pos['entry_price']
        market_value = pos['quantity'] * current_price
        pnl = market_value - cost_basis
        pnl_ratio = pnl / cost_basis

        signal_info = {
            'ts_code': ts_code,
            'prob': prob,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_ratio': pnl_ratio
        }

        if prob < sell_threshold:
            print(f"🔴 {ts_code}: 卖出信号！")
            print(f"   概率: {prob:.4f} < {sell_threshold}")
            print(f"   现价: {current_price:.2f} (成本: {pos['entry_price']:.2f})")
            print(f"   盈亏: {pnl:+.2f} ({pnl_ratio*100:+.2f}%)")
            sell_signals.append(signal_info)
        else:
            status = "🟢 持有" if prob > buy_threshold else "🟡 观望"
            print(f"{status} {ts_code}")
            print(f"   概率: {prob:.4f}")
            print(f"   现价: {current_price:.2f} (成本: {pos['entry_price']:.2f})")
            print(f"   盈亏: {pnl:+.2f} ({pnl_ratio*100:+.2f}%)")
            hold_signals.append(signal_info)
        print()

    print("=" * 60)
    print("总结")
    print("=" * 60)
    print(f"持仓数量: {len(positions)}")
    print(f"卖出信号: {len(sell_signals)} 个")
    print(f"持有/观望: {len(hold_signals)} 个")

    if sell_signals:
        print("\n建议卖出:")
        for sig in sell_signals:
            print(f"  - {sig['ts_code']}: 概率 {sig['prob']:.4f}, 现价 {sig['current_price']:.2f}")
    else:
        print("\n无卖出信号，继续持有")

    print("\n下次检查: 明天收盘后")


if __name__ == "__main__":
    main()
