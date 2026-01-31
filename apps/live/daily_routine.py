#!/usr/bin/env python3
"""
每日自动化交易引擎 v2.0

每天收盘后自动执行:
1. 检查是否为交易日
2. 检查所有持仓和新信号
3. 执行卖出/调仓操作
4. 计算当日盈亏
5. 更新状态文件
6. 记录健康检查

用法:
  - 手动运行: python3 scripts/daily_routine.py
  - 定时运行: 配置cron或systemd timer
"""

import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Tuple
import traceback
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor
from src.utils.trade_calendar import get_trade_calendar
import xgboost as xgb
from loguru import logger

# 导入统一配置
import config.settings as settings


class StateValidationError(Exception):
    """状态文件验证错误"""
    pass


class DailyTradingEngine:
    """每日自动交易引擎"""

    def __init__(self):
        """初始化引擎"""
        self.state_file = settings.get_state_file_path()
        self.model_file = settings.get_model_path()

        # 验证配置
        validation = settings.validate_config()
        if not validation['valid']:
            logger.error("配置验证失败:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
            raise ValueError("配置无效，请检查config/settings.py")

        # 加载模型
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(self.model_file))
            logger.info(f"模型加载成功: {self.model_file}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

        self.feature_extractor = EnhancedFeatureExtractor()
        self.storage = SQLiteStorage()
        self.trade_calendar = get_trade_calendar()

        # 获取股票池
        all_stocks = self.storage.get_all_stocks()
        self.universe = sorted(all_stocks)[:settings.UNIVERSE_SIZE]

        logger.info(f"交易引擎初始化完成")
        logger.info(f"  股票池: {len(self.universe)} 只")
        logger.info(f"  最大持仓: {settings.MAX_POSITIONS} 只")
        logger.info(f"  买入阈值: {settings.BUY_THRESHOLD}")
        logger.info(f"  卖出阈值: {settings.SELL_THRESHOLD}")

    def load_state(self) -> Optional[dict]:
        """加载状态文件"""
        if not self.state_file.exists():
            logger.error(f"状态文件不存在: {self.state_file}")
            logger.error("请先运行回测初始化")
            return None

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # 验证状态文件
            self._validate_state(state)
            return state

        except json.JSONDecodeError as e:
            logger.error(f"状态文件JSON格式错误: {e}")
            return None
        except StateValidationError as e:
            logger.error(f"状态文件验证失败: {e}")
            return None
        except Exception as e:
            logger.error(f"加载状态文件失败: {e}")
            return None

    def _validate_state(self, state: dict):
        """验证状态文件数据完整性"""
        required_fields = ['capital', 'positions', 'initial_capital']

        for field in required_fields:
            if field not in state:
                raise StateValidationError(f"缺少必需字段: {field}")

        # 验证capital
        if not isinstance(state['capital'], (int, float, str)):
            raise StateValidationError(f"capital类型错误: {type(state['capital'])}")

        try:
            capital = Decimal(str(state['capital']))
            if capital < 0:
                raise StateValidationError(f"capital不能为负数: {capital}")
        except:
            raise StateValidationError(f"capital值无效: {state['capital']}")

        # 验证positions
        if not isinstance(state['positions'], dict):
            raise StateValidationError(f"positions类型错误: {type(state['positions'])}")

        for ts_code, pos in state['positions'].items():
            if not isinstance(pos, dict):
                raise StateValidationError(f"position {ts_code} 类型错误")

            required_pos_fields = ['entry_date', 'entry_price', 'quantity']
            for field in required_pos_fields:
                if field not in pos:
                    raise StateValidationError(f"position {ts_code} 缺少字段: {field}")

            # 验证数值
            if pos['quantity'] <= 0 or pos['quantity'] % 100 != 0:
                raise StateValidationError(f"position {ts_code} 数量无效: {pos['quantity']}")

            if pos['entry_price'] <= 0:
                raise StateValidationError(f"position {ts_code} 价格无效: {pos['entry_price']}")

        logger.info("状态文件验证通过")

    def save_state(self, state: dict):
        """保存状态文件"""
        try:
            # 创建备份
            backup_dir = settings.STATE_BACKUP_DIR
            backup_dir.mkdir(exist_ok=True)

            backup_file = backup_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import shutil
            if self.state_file.exists():
                shutil.copy2(self.state_file, backup_file)
                logger.debug(f"状态文件已备份: {backup_file}")

            # 清理旧备份（保留最近7天）
            self._cleanup_old_backups(backup_dir)

            # 保存状态
            state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            logger.debug("状态文件已保存")

        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
            raise

    def _cleanup_old_backups(self, backup_dir: Path):
        """清理旧备份"""
        try:
            backups = sorted(backup_dir.glob('state_*.json'))
            if len(backups) > settings.MAX_BACKUPS:
                for old_backup in backups[:-settings.MAX_BACKUPS]:
                    old_backup.unlink()
                    logger.debug(f"删除旧备份: {old_backup}")
        except Exception as e:
            logger.warning(f"清理备份失败: {e}")

    def check_signal(self, ts_code: str, decision_date: str) -> Tuple[Optional[float], Optional[float]]:
        """
        检查单个股票的信号

        Returns:
            (概率, 当前价格) 或 (None, None)
        """
        try:
            end_date = pd.to_datetime(decision_date).strftime('%Y%m%d')
            start_date = (pd.to_datetime(decision_date) - timedelta(days=settings.FEATURE_LOOKBACK_DAYS)).strftime('%Y%m%d')

            df = self.storage.get_daily_prices(ts_code, start_date, end_date)

            if df is None or len(df) < settings.MIN_HISTORY_DAYS:
                return None, None

            features = self.feature_extractor.extract(df)

            if len(features) < 1:
                return None, None

            latest = features.iloc[-1]
            feature_cols = [c for c in features.columns if c.startswith('f_')]

            if len(feature_cols) == 0:
                return None, None

            X = latest[feature_cols].values.reshape(1, -1)
            prob = float(self.model.predict(xgb.DMatrix(X))[0])

            current_price = float(latest['close'])

            return prob, current_price

        except Exception as e:
            logger.warning(f"检查 {ts_code} 信号失败: {e}")
            return None, None

    def scan_market(self, decision_date: str, current_positions: dict) -> Tuple[List[dict], List[dict]]:
        """
        扫描全市场信号

        Returns:
            (买入信号列表, 持仓信号列表)
        """
        buy_signals = []
        hold_signals = []

        buy_threshold = float(settings.BUY_THRESHOLD)

        for i, ts_code in enumerate(self.universe):
            prob, price = self.check_signal(ts_code, decision_date)

            if prob is None:
                continue

            # 当前持仓
            if ts_code in current_positions:
                hold_signals.append({
                    'ts_code': ts_code,
                    'prob': prob,
                    'price': price
                })

            # 新买入信号
            elif prob > buy_threshold:
                buy_signals.append({
                    'ts_code': ts_code,
                    'prob': prob,
                    'price': price
                })

            if (i + 1) % 100 == 0:
                logger.debug(f"已扫描 {i+1}/{len(self.universe)} 只股票")

        # 排序
        buy_signals.sort(key=lambda x: x['prob'], reverse=True)
        hold_signals.sort(key=lambda x: x['prob'])

        return buy_signals, hold_signals

    def execute_trade(self, state: dict, ts_code: str, action: str, price: float, quantity: Optional[int] = None) -> Optional[dict]:
        """
        执行交易

        Returns:
            交易记录或None
        """
        capital = Decimal(str(state['capital']))
        positions = state['positions']
        trades = state.get('trades', [])
        trade_date = datetime.now().strftime('%Y-%m-%d')

        if action == 'sell':
            if ts_code not in positions:
                logger.warning(f"卖出失败: {ts_code} 不在持仓中")
                return None

            pos = positions[ts_code]
            qty = int(pos['quantity'])
            entry_price = Decimal(str(pos['entry_price']))

            # 卖出（使用Decimal计算）
            sell_value = Decimal(str(qty * price)) * (Decimal('1') - settings.SELL_COST_RATE)
            pnl = sell_value - (Decimal(str(qty)) * entry_price)
            pnl_ratio = float(pnl / (Decimal(str(qty)) * entry_price))

            state['capital'] = float(sell_value + capital)
            del positions[ts_code]

            trade = {
                "date": trade_date,
                "ts_code": ts_code,
                "action": "sell",
                "price": float(price),
                "quantity": qty,
                "amount": float(sell_value),
                "pnl": float(pnl),
                "pnl_ratio": pnl_ratio,
                "capital_after": state['capital']
            }
            trades.append(trade)
            state['trades'] = trades

            logger.info(f"卖出 {ts_code}: {qty:,}股 @ {price:.2f}, 盈亏={pnl:+.2f} ({pnl_ratio*100:+.2f}%)")
            return trade

        elif action == 'buy':
            if ts_code in positions:
                logger.warning(f"买入失败: {ts_code} 已在持仓中")
                return None

            if len(positions) >= settings.MAX_POSITIONS:
                logger.warning(f"买入失败: 持仓已满 ({settings.MAX_POSITIONS})")
                return None

            # 计算买入数量
            buy_value = capital * settings.POSITION_SIZE
            qty = int(buy_value / Decimal(str(price)) / 100) * 100

            if qty < 100:
                logger.warning(f"买入失败: 资金不足 (可买{qty}股)")
                return None

            cost = Decimal(str(qty * price)) * (Decimal('1') + settings.BUY_COST_RATE)

            if cost > capital:
                logger.warning(f"买入失败: 资金不足 (需要{cost:.2f}, 可用{capital:.2f})")
                return None

            # 执行买入
            state['capital'] = float(capital - cost)
            positions[ts_code] = {
                "entry_date": trade_date,
                "entry_price": float(price),
                "quantity": qty
            }

            trade = {
                "date": trade_date,
                "ts_code": ts_code,
                "action": "buy",
                "price": float(price),
                "quantity": qty,
                "amount": float(cost),
                "capital_after": state['capital']
            }
            trades.append(trade)
            state['trades'] = trades

            logger.info(f"买入 {ts_code}: {qty:,}股 @ {price:.2f}, 成本={cost:.2f}")
            return trade

        return None

    def calculate_pnl(self, state: dict, decision_date: str) -> dict:
        """计算当日盈亏"""
        capital = Decimal(str(state['capital']))
        positions = state['positions']

        total_market_value = Decimal('0')
        position_details = []

        for ts_code, pos in positions.items():
            try:
                df = self.storage.get_daily_prices(ts_code, decision_date, decision_date)
                if df is None or df.empty:
                    current_price = Decimal(str(pos['entry_price']))
                else:
                    current_price = Decimal(str(df['close'].iloc[-1]))

                qty = Decimal(str(pos['quantity']))
                entry_price = Decimal(str(pos['entry_price']))

                market_value = qty * current_price
                cost_basis = qty * entry_price
                pnl = market_value - cost_basis
                pnl_ratio = float(pnl / cost_basis) if cost_basis > 0 else 0

                total_market_value += market_value

                position_details.append({
                    'ts_code': ts_code,
                    'quantity': pos['quantity'],
                    'entry_price': float(entry_price),
                    'current_price': float(current_price),
                    'market_value': float(market_value),
                    'pnl': float(pnl),
                    'pnl_ratio': pnl_ratio
                })

            except Exception as e:
                logger.warning(f"计算 {ts_code} 盈亏失败: {e}")

        total_value = capital + total_market_value
        initial_capital = Decimal(str(state.get('initial_capital', 1000000)))
        total_return = float((total_value - initial_capital) / initial_capital)

        return {
            'capital': float(capital),
            'market_value': float(total_market_value),
            'total_value': float(total_value),
            'total_return': total_return,
            'positions': position_details
        }

    def write_heartbeat(self, status: str, message: str = ""):
        """写入健康检查心跳"""
        try:
            settings.HEARTBEAT_FILE.parent.mkdir(exist_ok=True)
            with open(settings.HEARTBEAT_FILE, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {status}: {message}\n")
        except Exception as e:
            logger.warning(f"写入心跳失败: {e}")

    def run(self, decision_date: str = None, force_run: bool = False):
        """
        执行每日例程

        Args:
            decision_date: 决策日期 (YYYYMMDD格式)
            force_run: 强制运行，跳过交易日检查
        """
        logger.info("=" * 60)
        logger.info(f"每日交易例程 v2.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # 1. 检查是否应该运行
        if not force_run:
            should_run, reason = self.trade_calendar.should_run_daily_routine()
            if not should_run:
                logger.info(f"跳过执行: {reason}")
                self.write_heartbeat("SKIPPED", reason)
                return

        # 2. 加载状态
        state = self.load_state()
        if state is None:
            logger.error("无法加载状态文件，退出")
            self.write_heartbeat("ERROR", "状态文件加载失败")
            return

        # 3. 确定决策日期
        if decision_date is None:
            decision_date = self.trade_calendar.get_latest_trading_date()
            if decision_date is None:
                logger.error("无法获取最新交易日")
                self.write_heartbeat("ERROR", "无法获取交易日")
                return

            today = datetime.now().strftime('%Y%m%d')
            if decision_date != today:
                logger.warning(f"今日数据尚未更新，使用最新交易日: {decision_date}")
            else:
                logger.info(f"使用今日数据: {decision_date}")

        logger.info(f"决策日期: {decision_date}")
        logger.info(f"当前现金: {settings.format_money(state['capital'])} 元")
        logger.info(f"当前持仓: {len(state['positions'])} 只")

        try:
            # 4. 扫描市场
            logger.info("\n扫描市场信号...")
            buy_signals, hold_signals = self.scan_market(decision_date, state['positions'])

            logger.info(f"买入信号: {len(buy_signals)} 个")
            logger.info(f"持有信号: {len(hold_signals)} 个")

            # 5. 检查卖出信号
            logger.info("\n" + "-" * 60)
            logger.info("第一步: 检查卖出信号")
            logger.info("-" * 60)

            sell_threshold = float(settings.SELL_THRESHOLD)
            for hold in hold_signals:
                if hold['prob'] < sell_threshold:
                    self.execute_trade(state, hold['ts_code'], 'sell', hold['price'])

            # 6. 检查调仓信号
            logger.info("\n" + "-" * 60)
            logger.info("第二步: 检查调仓信号")
            logger.info("-" * 60)

            hold_signals = [h for h in hold_signals if h['ts_code'] in state['positions']]
            hold_signals.sort(key=lambda x: x['prob'])

            if hold_signals and buy_signals:
                weakest = hold_signals[0]
                strongest = buy_signals[0]

                rebalance_threshold = float(settings.REBALANCE_THRESHOLD)
                if strongest['prob'] > weakest['prob'] + rebalance_threshold:
                    logger.info(f"调仓: 卖出 {weakest['ts_code']} (概率={weakest['prob']:.4f})")
                    logger.info(f"      买入 {strongest['ts_code']} (概率={strongest['prob']:.4f})")

                    # 卖出最弱持仓
                    self.execute_trade(state, weakest['ts_code'], 'sell', weakest['price'])

                    # 买入最强信号
                    hold_signals = [h for h in hold_signals if h['ts_code'] in state['positions']]
                    if len(state['positions']) < settings.MAX_POSITIONS:
                        self.execute_trade(state, strongest['ts_code'], 'buy', strongest['price'])

            # 7. 检查新买入
            logger.info("\n" + "-" * 60)
            logger.info("第三步: 检查新买入")
            logger.info("-" * 60)

            # 重新获取买入信号（排除已持仓）
            buy_signals, _ = self.scan_market(decision_date, state['positions'])

            for buy in buy_signals:
                if len(state['positions']) < settings.MAX_POSITIONS:
                    self.execute_trade(state, buy['ts_code'], 'buy', buy['price'])
                else:
                    break

            # 8. 计算当日盈亏
            logger.info("\n" + "-" * 60)
            logger.info("第四步: 计算当日盈亏")
            logger.info("-" * 60)

            pnl_info = self.calculate_pnl(state, decision_date)

            logger.info(f"现金: {settings.format_money(pnl_info['capital'])} 元")
            logger.info(f"持仓市值: {settings.format_money(pnl_info['market_value'])} 元")
            logger.info(f"总资产: {settings.format_money(pnl_info['total_value'])} 元")
            logger.info(f"总收益率: {pnl_info['total_return']*100:.2f}%")

            for pos in pnl_info['positions']:
                logger.info(f"  {pos['ts_code']}: {pos['quantity']:,}股, "
                           f"盈亏={pos['pnl']:+.2f} ({pos['pnl_ratio']*100:+.2f}%)")

            # 9. 保存当日净值
            daily_values = state.get('daily_values', [])

            # 避免重复记录同一天
            if not daily_values or daily_values[-1].get('date') != decision_date:
                daily_values.append({
                    'date': decision_date,
                    'total_value': pnl_info['total_value'],
                    'capital': pnl_info['capital'],
                    'positions': len(state['positions']),
                    'market_value': pnl_info['market_value']
                })
            else:
                # 更新最后一条记录
                daily_values[-1] = {
                    'date': decision_date,
                    'total_value': pnl_info['total_value'],
                    'capital': pnl_info['capital'],
                    'positions': len(state['positions']),
                    'market_value': pnl_info['market_value']
                }

            state['daily_values'] = daily_values

            # 10. 保存状态
            self.save_state(state)

            logger.info("\n" + "=" * 60)
            logger.info("每日例程完成")
            logger.info("=" * 60)

            self.write_heartbeat("SUCCESS", f"总资产: {settings.format_money(pnl_info['total_value'])}")

        except Exception as e:
            logger.error(f"执行例程失败: {e}")
            logger.error(traceback.format_exc())
            self.write_heartbeat("ERROR", str(e))


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, format=settings.LOG_FORMAT, level=settings.LOG_LEVEL)

    # 文件日志
    log_file = settings.get_log_file("daily_trading")
    log_file.parent.mkdir(exist_ok=True)
    logger.add(log_file, rotation=settings.LOG_ROTATION, retention=settings.LOG_RETENTION, level="DEBUG")

    # 检查是否强制运行
    force_run = '--force' in sys.argv or '-f' in sys.argv

    try:
        engine = DailyTradingEngine()
        engine.run(force_run=force_run)
    except Exception as e:
        logger.error(f"系统错误: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
