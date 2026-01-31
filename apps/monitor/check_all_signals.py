#!/usr/bin/env python3
"""
检查全市场交易信号

检查当前持仓的卖出信号 + 全市场的买入信号
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
from loguru import logger


def check_signal(ts_code, decision_date, storage, model, feature_extractor):
    """检查单个股票的信号"""
    try:
        end_date = pd.to_datetime(decision_date).strftime('%Y%m%d')
        start_date = (pd.to_datetime(decision_date) - timedelta(days=120)).strftime('%Y%m%d')

        df = storage.get_daily_prices(ts_code, start_date, end_date)

        if df is None or len(df) < 60:
            return None, None, None

        features = feature_extractor.extract(df)

        if len(features) < 1:
            return None, None, None

        latest = features.iloc[-1]
        feature_cols = [c for c in features.columns if c.startswith('f_')]

        if len(feature_cols) == 0:
            return None, None, None

        X = latest[feature_cols].values.reshape(1, -1)
        prob = model.predict(xgb.DMatrix(X))[0]

        current_price = latest['close']
        trade_date = latest['trade_date']

        return prob, current_price, trade_date

    except Exception as e:
        logger.warning(f"检查 {ts_code} 信号失败: {e}")
        return None, None, None


def main():
    """检查全市场信号"""
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")

    print("=" * 70)
    print("全市场交易信号检查")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 当前持仓
    current_positions = {
        "000536.SZ": {"entry_price": 4.08, "quantity": 73500},
        "000586.SZ": {"entry_price": 8.51, "quantity": 24600},
        "000548.SZ": {"entry_price": 4.88, "quantity": 30100},
    }

    # 策略参数
    MAX_POSITIONS = 3
    BUY_THRESHOLD = 0.52
    SELL_THRESHOLD = 0.48

    # 决策日期
    decision_date = '20250129'

    # 加载模型
    model = xgb.Booster()
    model.load_model('models/xgboost_2022_2026.json')
    feature_extractor = EnhancedFeatureExtractor()
    storage = SQLiteStorage()

    # 获取股票池（前500只）
    universe = sorted(storage.get_all_stocks())[:500]

    print(f"\n参数设置:")
    print(f"  决策日期: {decision_date}")
    print(f"  买入阈值: 概率 > {BUY_THRESHOLD}")
    print(f"  卖出阈值: 概率 < {SELL_THRESHOLD}")
    print(f"  最大持仓: {MAX_POSITIONS} 只")
    print(f"  当前持仓: {len(current_positions)} 只")
    print(f"  股票池: {len(universe)} 只\n")

    print("扫描市场信号...")

    buy_signals = []      # 新的买入信号
    sell_signals = []     # 当前持仓的卖出信号
    hold_signals = []     # 当前持仓的持有信号

    # 检查所有股票
    for i, ts_code in enumerate(universe):
        prob, price, date = check_signal(ts_code, decision_date, storage, model, feature_extractor)

        if prob is None:
            continue

        # 是当前持仓
        if ts_code in current_positions:
            pos = current_positions[ts_code]
            cost_basis = pos['quantity'] * pos['entry_price']
            market_value = pos['quantity'] * price
            pnl = market_value - cost_basis
            pnl_ratio = pnl / cost_basis

            if prob < SELL_THRESHOLD:
                sell_signals.append({
                    'ts_code': ts_code,
                    'prob': prob,
                    'price': price,
                    'pnl': pnl,
                    'pnl_ratio': pnl_ratio
                })
            else:
                hold_signals.append({
                    'ts_code': ts_code,
                    'prob': prob,
                    'price': price,
                    'pnl': pnl,
                    'pnl_ratio': pnl_ratio
                })

        # 不是持仓，但有买入信号
        elif prob > BUY_THRESHOLD:
            buy_signals.append({
                'ts_code': ts_code,
                'prob': prob,
                'price': price
            })

        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  已扫描 {i+1}/{len(universe)} 只股票...")

    print(f"\n扫描完成！")

    # 排序买入信号（按概率降序）
    buy_signals.sort(key=lambda x: x['prob'], reverse=True)

    # 显示结果
    print("\n" + "=" * 70)
    print("当前持仓状态")
    print("=" * 70)

    if sell_signals:
        print(f"\n🔴 卖出信号 ({len(sell_signals)} 个):")
        for sig in sell_signals:
            print(f"  {sig['ts_code']}: 概率={sig['prob']:.4f}, 现价={sig['price']:.2f}, "
                  f"盈亏={sig['pnl']:+.2f} ({sig['pnl_ratio']*100:+.2f}%)")

    if hold_signals:
        print(f"\n🟢 持有信号 ({len(hold_signals)} 个):")
        for sig in hold_signals:
            print(f"  {sig['ts_code']}: 概率={sig['prob']:.4f}, 现价={sig['price']:.2f}, "
                  f"盈亏={sig['pnl']:+.2f} ({sig['pnl_ratio']*100:+.2f}%)")

    print("\n" + "=" * 70)
    print(f"新买入信号 (共 {len(buy_signals)} 个)")
    print("=" * 70)

    if buy_signals:
        print("\n前20个买入信号:")
        print(f"{'股票代码':<12} {'概率':<10} {'现价':<10}")
        print("-" * 35)
        for sig in buy_signals[:20]:
            print(f"{sig['ts_code']:<12} {sig['prob']:.4f}     {sig['price']:.2f}")

        # 对比当前持仓
        print("\n" + "-" * 70)
        print("信号对比分析:")
        print("-" * 70)

        print("\n当前持仓 vs 最强买入信号:")
        for i, hold in enumerate(hold_signals):
            print(f"\n持仓{i+1}: {hold['ts_code']} (概率={hold['prob']:.4f})")

            # 找出比它强的新信号
            stronger = [b for b in buy_signals if b['prob'] > hold['prob']]
            if stronger:
                print(f"  有 {len(stronger)} 个更强的买入信号")
                for sig in stronger[:3]:
                    print(f"    - {sig['ts_code']}: 概率={sig['prob']:.4f} (更强 {((sig['prob']/hold['prob']-1)*100):.1f}%)")

    # 决策建议
    print("\n" + "=" * 70)
    print("操作建议")
    print("=" * 70)

    action_needed = False

    # 1. 有卖出信号？
    if sell_signals:
        print(f"\n⚠️  有 {len(sell_signals)} 个持仓出现卖出信号，建议卖出:")
        for sig in sell_signals:
            print(f"    - {sig['ts_code']}: 概率 {sig['prob']:.4f} < {SELL_THRESHOLD}")
        action_needed = True

    # 2. 有更强的买入信号且仓位已满？
    elif len(hold_signals) >= MAX_POSITIONS and buy_signals:
        # 找出最弱持仓
        weakest_hold = min(hold_signals, key=lambda x: x['prob'])
        strongest_buy = buy_signals[0]

        if strongest_buy['prob'] > weakest_hold['prob'] + 0.05:  # 概率差异超过5%
            print(f"\n⚠️  发现更强的买入信号，建议调仓:")
            print(f"    卖出: {weakest_hold['ts_code']} (概率={weakest_hold['prob']:.4f})")
            print(f"    买入: {strongest_buy['ts_code']} (概率={strongest_buy['prob']:.4f})")
            action_needed = True
        else:
            print(f"\n✅ 无需操作")
            print(f"    当前持仓概率均较高，虽有新买入信号但优势不明显")
            print(f"    最强新信号: {strongest_buy['ts_code']} (概率={strongest_buy['prob']:.4f})")
            print(f"    最弱持仓: {weakest_hold['ts_code']} (概率={weakest_hold['prob']:.4f})")

    # 3. 有空位且有买入信号？
    elif len(hold_signals) < MAX_POSITIONS and buy_signals:
        slots_available = MAX_POSITIONS - len(hold_signals)
        print(f"\n⚠️  有 {len(buy_signals)} 个买入信号，可用仓位 {slots_available} 个")
        print(f"  建议买入:")
        for i, sig in enumerate(buy_signals[:slots_available]):
            print(f"    {i+1}. {sig['ts_code']}: 概率={sig['prob']:.4f}, 现价={sig['price']:.2f}")
        action_needed = True

    else:
        print(f"\n✅ 无需操作")
        print(f"    当前持仓: {len(hold_signals)} 只")
        print(f"    新买入信号: {len(buy_signals)} 个")
        if buy_signals:
            print(f"    最强新信号: {buy_signals[0]['ts_code']} (概率={buy_signals[0]['prob']:.4f})")

    print("\n" + "=" * 70)
    print(f"下次检查: 明天收盘后")
    print("=" * 70)


if __name__ == "__main__":
    main()
