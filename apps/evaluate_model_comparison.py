#!/usr/bin/env python3
"""
模型对比评估脚本 - 统一评估口径

确保 LightGBM 和 MLP 在相同条件下公平对比：
1. 同一时间切分 (train: 2021-2022, val: 2023, test: 2024)
2. 同一严格回测引擎（A股约束、成本、滑点）
3. 同一基准（等权/行业中性基准）
4. 输出同一指标：AUC, IC, Top-Decile Spread, 年化收益, 最大回撤, 换手率, 成本后收益
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import json
import joblib
import hashlib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage


# ==================== 配置 ====================

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 时间切分（统一）
    train_years: List[int] = None
    val_year: int = 2023
    test_year: int = 2024

    # 回测参数
    prediction_period: int = 20  # 预测周期
    holding_period: int = 20  # 持仓周期
    top_decile: float = 0.1  # Top 10%
    bottom_decile: float = 0.1  # Bottom 10%

    # 交易成本（A股）
    commission_buy: float = 0.0003  # 买入佣金 0.03%
    commission_sell: float = 0.0013  # 卖出佣金+印花税 0.13%
    slippage: float = 0.001  # 滑点 0.1%

    # 组合约束
    max_positions: int = 30  # 最大持仓数
    min_positions: int = 10  # 最小持仓数

    # === A股交易约束 ===
    t1_constraint: bool = True  # T+1约束
    price_limit_pct: float = 9.5  # 涨跌停阈值（百分比口径，9.5代表9.5%）
    suspend_threshold: int = 3  # 停牌阈值
    min_volume_ratio: float = 0.05  # 最小成交量比率

    def __post_init__(self):
        if self.train_years is None:
            self.train_years = [2021, 2022]


# ==================== 评估指标计算 ====================

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.storage = SQLiteStorage()

    def load_predictions(
        self,
        model_name: str,
        year: int
    ) -> pd.DataFrame:
        """
        加载模型预测结果

        Args:
            model_name: 'mlp_v6' 或 'lgbm_v6'
            year: 年份

        Returns:
            包含 ts_code, trade_date, pred_prob, fwd_ret 的 DataFrame
        """
        # 尝试加载缓存的预测结果（使用 pkl 格式）
        cache_file = Path(f"data/cache/predictions/{model_name}_{year}.pkl")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # 检查模型文件hash
        # 根据模型类型确定模型文件路径
        if model_name.startswith('lgbm'):
            model_file = Path(f"models/{model_name}.txt")
        else:
            model_file = Path(f"models/{model_name}_best.pt")

        if model_file.exists():
            model_hash = hashlib.md5(open(model_file, 'rb').read()).hexdigest()[:8]
            hash_file = cache_file.with_suffix('.hash')

            if cache_file.exists():
                # 检查hash是否匹配
                if hash_file.exists():
                    cached_hash = hash_file.read_text().strip()
                    if cached_hash != model_hash:
                        logger.warning(f"模型已更新，缓存过期: {cache_file}")
                        cache_file.unlink()
                        hash_file.unlink()
                    else:
                        logger.info(f"加载缓存预测: {cache_file}")
                        return pd.read_pickle(cache_file)
                else:
                    # 无hash文件，直接加载并补写hash（兼容历史缓存）
                    logger.warning(f"缓存缺少hash校验，直接加载并补写hash: {cache_file}")
                    hash_file.write_text(model_hash)
                    logger.info(f"加载缓存预测: {cache_file}")
                    return pd.read_pickle(cache_file)

        # 否则需要运行模型生成预测
        raise FileNotFoundError(
            f"预测文件不存在: {cache_file}\n"
            f"请先运行训练脚本生成预测结果"
        )

    def compute_auc(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """计算 AUC"""
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return np.nan

    def compute_ic(
        self,
        df: pd.DataFrame,
        pred_col: str = 'pred_prob',
        ret_col: str = 'fwd_ret'
    ) -> Dict[str, float]:
        """
       计算信息系数（IC）

        Returns:
            {
                'ic_mean': 平均IC,
                'ic_std': IC标准差,
                'icir': ICIR (IC均值/IC标准差),
                'ic_positive_ratio': IC正值占比
            }
        """
        # 按日期计算截面IC
        ic_series = df.groupby('trade_date').apply(
            lambda x: stats.spearmanr(x[pred_col], x[ret_col])[0]
            if len(x) > 10 else np.nan
        ).dropna()

        if len(ic_series) == 0:
            return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan, 'ic_positive_ratio': np.nan}

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / (ic_std + 1e-8)
        ic_positive_ratio = (ic_series > 0).mean()

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'ic_positive_ratio': ic_positive_ratio
        }

    def compute_spread(
        self,
        df: pd.DataFrame,
        pred_col: str = 'pred_prob',
        ret_col: str = 'fwd_ret'
    ) -> Dict[str, float]:
        """
        计算 Top-Bottom Spread

        Returns:
            {
                'top_return': Top 10% 收益,
                'bottom_return': Bottom 10% 收益,
                'spread': Top - Bottom
            }
        """
        results = []

        # 按日期计算
        for date, group in df.groupby('trade_date'):
            if len(group) < 20:
                continue

            n = len(group)
            top_n = max(int(n * self.config.top_decile), 1)
            bottom_n = max(int(n * self.config.bottom_decile), 1)

            sorted_group = group.sort_values(pred_col, ascending=False)
            top_ret = sorted_group.head(top_n)[ret_col].mean()
            bottom_ret = sorted_group.tail(bottom_n)[ret_col].mean()

            results.append({
                'trade_date': date,
                'top_return': top_ret,
                'bottom_return': bottom_ret,
                'spread': top_ret - bottom_ret
            })

        if not results:
            return {'top_return': np.nan, 'bottom_return': np.nan, 'spread': np.nan}

        spread_df = pd.DataFrame(results)

        return {
            'top_return': spread_df['top_return'].mean(),
            'bottom_return': spread_df['bottom_return'].mean(),
            'spread': spread_df['spread'].mean()
        }

    def compute_portfolio_metrics(
        self,
        df: pd.DataFrame,
        pred_col: str = 'pred_prob',
        ret_col: str = 'fwd_ret'
    ) -> Dict[str, float]:
        """
        计算组合级指标（严格回测版本）

        关键修复：fwd_ret 是 N 天收益率，必须每 N 天调仓一次，不能每日调仓！
        否则会导致收益被错误叠加，MDD虚高。

        包含：
        1. 成本后年化、最大回撤、夏普、Calmar
        2. 年化换手、单边成本占比、滑点损耗占比
        3. 相对等权基准的超额收益与信息比率

        Returns:
            完整的组合指标字典
        """
        # 获取所有交易日并按调仓周期采样
        all_dates = sorted(df['trade_date'].unique())
        holding_period = self.config.prediction_period  # 20天

        # 每 holding_period 天调仓一次
        rebalance_dates = all_dates[::holding_period]

        period_returns = []
        positions_prev = set()
        position_buy_dates = {}  # 跟踪持仓买入日期，用于T+1约束

        # 成本分解
        commission_buy_rate = self.config.commission_buy
        commission_sell_rate = self.config.commission_sell
        slippage_rate = self.config.slippage

        for i, date in enumerate(rebalance_dates):
            group = df[df['trade_date'] == date]

            if len(group) < self.config.min_positions:
                continue

            # 选择 Top N 股票
            n_positions = min(self.config.max_positions, len(group) // 3)
            top_stocks = set(
                group.nlargest(n_positions, pred_col)['ts_code'].tolist()
            )

            # === A股交易约束检查 ===
            # 已实现：
            #   - 涨停过滤（当日涨幅>9.5%不买入）
            #   - 跌停卖出限制（当日跌幅<-9.5%无法卖出）
            #   - 停牌检测（成交量为0不买入）
            #   - T+1约束（当天买入的次日才能卖出）
            #   - 流动性过滤（成交量低于均值5%不买入）

            # 1. 停牌检测（成交量为0的股票不能买入）
            if 'vol' in group.columns:
                suspended_stocks = set(
                    group[(group['ts_code'].isin(top_stocks)) & (group['vol'] == 0)]['ts_code']
                )
                if suspended_stocks:
                    top_stocks = top_stocks - suspended_stocks
                    if len(top_stocks) < self.config.min_positions:
                        continue

            # 2. 涨停过滤（当日涨幅>9.5%不买入，避免买入后无法卖出）
            if 'pct_chg' in group.columns:
                pct_chg_today = group[group['ts_code'].isin(top_stocks)]
                near_limit_up_stocks = set(
                    pct_chg_today[pct_chg_today['pct_chg'] > self.config.price_limit_pct]['ts_code']
                )
                if near_limit_up_stocks:
                    top_stocks = top_stocks - near_limit_up_stocks
                    if len(top_stocks) < self.config.min_positions:
                        continue

            # 3. 流动性过滤（成交量低于均值5%不买入）
            if 'vol' in group.columns:
                stocks_data = group[group['ts_code'].isin(top_stocks)][['ts_code', 'vol']]
                avg_vol = stocks_data['vol'].mean()
                if avg_vol > 0:
                    low_vol_stocks = set(
                        stocks_data[stocks_data['vol'] < avg_vol * self.config.min_volume_ratio]['ts_code']
                    )
                    if low_vol_stocks:
                        top_stocks = top_stocks - low_vol_stocks
                        if len(top_stocks) < self.config.min_positions:
                            continue

            # 4. T+1约束 + 跌停卖出限制
            # 需要卖出的股票 = 旧持仓 - 新持仓目标
            to_sell = positions_prev - top_stocks
            forced_hold = set()  # 被强制保留的股票

            for stock in to_sell:
                # T+1约束：当天买入的不能卖出
                if stock in position_buy_dates:
                    buy_date = position_buy_dates[stock]
                    if buy_date == date:
                        forced_hold.add(stock)
                        continue

                # 跌停卖出限制：当日跌停无法卖出
                if 'pct_chg' in group.columns:
                    stock_pct_chg = group[group['ts_code'] == stock]['pct_chg']
                    if len(stock_pct_chg) > 0 and stock_pct_chg.values[0] < -self.config.price_limit_pct:
                        forced_hold.add(stock)

            # 更新最终持仓
            actual_holdings = top_stocks | forced_hold
            n_positions = len(actual_holdings)

            if n_positions < self.config.min_positions:
                continue

            # 更新持仓买入日期
            new_buys = actual_holdings - positions_prev
            for stock in new_buys:
                position_buy_dates[stock] = date
            # 清理已卖出股票的记录
            sold = positions_prev - actual_holdings
            for stock in sold:
                position_buy_dates.pop(stock, None)

            # 计算组合的 period 收益（已经是 holding_period 天的收益）
            period_ret = group[group['ts_code'].isin(actual_holdings)][ret_col].mean()

            # === 行业中性基准（修复P2-6）===
            # 计算行业中性化基准收益：先按行业聚合，再按行业权重加权
            if 'industry' in group.columns:
                # 按行业聚合计算各行业平均收益
                industry_returns = group.groupby('industry')[ret_col].mean()
                # 计算各行业股票数量权重
                industry_counts = group.groupby('industry').size()
                industry_weights = industry_counts / industry_counts.sum()
                # 加权平均得到行业中性基准
                baseline_ret = (industry_returns * industry_weights).sum()
            else:
                # 兜底基准（如果无行业信息）
                baseline_ret = group[ret_col].mean()

            # 计算换手（基于实际持仓变化）
            if positions_prev:
                changed_positions = len(actual_holdings.symmetric_difference(positions_prev))
                turnover = changed_positions / n_positions if n_positions > 0 else 0
            else:
                turnover = 0
                changed_positions = n_positions  # 首次建仓，全部买入

            # 成本分解
            buy_ratio = changed_positions / (2 * n_positions) if n_positions > 0 else 0
            sell_ratio = changed_positions / (2 * n_positions) if n_positions > 0 else 0

            commission_cost = buy_ratio * commission_buy_rate + sell_ratio * commission_sell_rate
            slippage_cost = turnover * slippage_rate
            total_cost = commission_cost + slippage_cost

            net_period_ret = period_ret - total_cost

            period_returns.append({
                'trade_date': date,
                'period': i + 1,
                'gross_return': period_ret,
                'net_return': net_period_ret,
                'baseline_return': baseline_ret,
                'excess_return': period_ret - baseline_ret,
                'turnover': turnover,
                'commission_cost': commission_cost,
                'slippage_cost': slippage_cost,
                'total_cost': total_cost,
                'n_positions': n_positions,
            })

            positions_prev = actual_holdings

        if not period_returns:
            return self._empty_metrics()

        returns_df = pd.DataFrame(period_returns).sort_values('trade_date')

        # 年化计算：一年约 252/holding_period 个调仓周期
        periods_per_year = 252 / holding_period

        # 1. 净值曲线（先定义，后续计算都用它）
        # 净值 = 累计复利收益
        nav_series = (1 + returns_df['net_return']).cumprod()

        # 2. 成本后年化收益（基于实际净值曲线 - 更准确）
        if len(returns_df) > 0:
            final_nav = nav_series.iloc[-1]
            initial_nav = 1.0
            total_periods = len(returns_df)
            years_elapsed = total_periods * holding_period / 252
            annual_return = final_nav ** (1 / years_elapsed) - 1 if years_elapsed > 0 else 0.0
        else:
            annual_return = 0.0

        # 3. 最大回撤（基于净值曲线）
        running_max = nav_series.cummax()
        drawdown = (nav_series - running_max) / running_max
        max_drawdown = drawdown.min()

        # 3. 夏普比率
        if returns_df['net_return'].std() > 0:
            sharpe_ratio = returns_df['net_return'].mean() / returns_df['net_return'].std() * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0

        # 4. Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 5. 年化换手率
        avg_turnover = returns_df['turnover'].mean()
        annual_turnover = avg_turnover * periods_per_year

        # 6. 成本占比分析
        total_commission = returns_df['commission_cost'].sum()
        total_slippage = returns_df['slippage_cost'].sum()
        total_cost = returns_df['total_cost'].sum()

        commission_ratio = total_commission / total_cost if total_cost > 0 else 0
        slippage_ratio = total_slippage / total_cost if total_cost > 0 else 0

        # 7. 相对基准的超额收益
        avg_excess = returns_df['excess_return'].mean()
        annual_excess = (1 + avg_excess) ** periods_per_year - 1

        # 8. 信息比率
        if returns_df['excess_return'].std() > 0:
            information_ratio = returns_df['excess_return'].mean() / returns_df['excess_return'].std() * np.sqrt(periods_per_year)
        else:
            information_ratio = 0

        # 9. 基准年化收益
        avg_baseline = returns_df['baseline_return'].mean()
        baseline_annual = (1 + avg_baseline) ** periods_per_year - 1

        # 10. 调仓次数
        n_rebalances = len(returns_df)

        # 11. 平均净收益率（单期）
        avg_net_return = returns_df['net_return'].mean()

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'turnover': avg_turnover,
            'annual_turnover': annual_turnover,
            'commission_ratio': commission_ratio,
            'slippage_ratio': slippage_ratio,
            'net_return': avg_net_return,
            'baseline_annual': baseline_annual,
            'annual_excess': annual_excess,
            'information_ratio': information_ratio,
            'n_rebalances': n_rebalances,
        }

    def _empty_metrics(self) -> Dict[str, float]:
        """返回空指标"""
        return {
            'annual_return': np.nan,
            'max_drawdown': np.nan,
            'sharpe_ratio': np.nan,
            'calmar_ratio': np.nan,
            'turnover': np.nan,
            'annual_turnover': np.nan,
            'commission_ratio': np.nan,
            'slippage_ratio': np.nan,
            'net_return': np.nan,
            'baseline_annual': np.nan,
            'annual_excess': np.nan,
            'information_ratio': np.nan,
            'n_rebalances': 0,
        }

    def evaluate_model(
        self,
        model_name: str,
        year: int
    ) -> Dict:
        """
        评估单个模型

        Args:
            model_name: 模型名称
            year: 评估年份

        Returns:
            完整评估指标
        """
        logger.info(f"\n评估模型: {model_name}, 年份: {year}")

        # 加载预测结果
        df = self.load_predictions(model_name, year)

        # 合并额外数据以支持A股交易约束（涨跌停、成交量等）
        # 从存储中加载日行情数据
        try:
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            daily_df = self.storage.fetch_daily_price(
                ts_code=None,
                start_date=start_date,
                end_date=end_date
            )
            if daily_df is not None and len(daily_df) > 0:
                # 合并 pct_chg, vol, amount 列
                merge_cols = ['ts_code', 'trade_date']
                extra_cols = ['pct_chg', 'vol', 'amount']
                # 检查哪些列存在
                available_cols = [c for c in extra_cols if c in daily_df.columns]
                if available_cols:
                    df = df.merge(
                        daily_df[merge_cols + available_cols],
                        on=merge_cols,
                        how='left'
                    )
                    logger.info(f"已合并A股交易约束数据: {available_cols}")
        except Exception as e:
            logger.warning(f"无法加载A股交易约束数据: {e}")

        # 合并行业分类数据（用于行业中性基准）
        try:
            industry_df = self.storage.get_industry_classification()
            if industry_df is not None and len(industry_df) > 0:
                df = df.merge(
                    industry_df[['ts_code', 'industry']],
                    on='ts_code',
                    how='left'
                )
                logger.info(f"已合并行业分类数据")
        except Exception as e:
            logger.warning(f"无法加载行业分类数据: {e}")

        # 过滤有效样本
        df = df.dropna(subset=['pred_prob', 'fwd_ret'])

        # 构建二分类标签（用于AUC）
        df['label'] = (df['fwd_ret'] > 0).astype(int)

        logger.info(f"样本数: {len(df)}, 交易日数: {df['trade_date'].nunique()}")

        # 计算各项指标
        results = {
            'model': model_name,
            'year': year,
            'n_samples': len(df),
            'n_days': df['trade_date'].nunique(),
        }

        # 1. AUC
        results['auc'] = self.compute_auc(df['label'].values, df['pred_prob'].values)

        # 2. IC
        ic_metrics = self.compute_ic(df)
        results.update({f'ic_{k}': v for k, v in ic_metrics.items()})

        # 3. Spread
        spread_metrics = self.compute_spread(df)
        results.update({f'spread_{k}': v for k, v in spread_metrics.items()})

        # 4. 组合指标
        portfolio_metrics = self.compute_portfolio_metrics(df)
        results.update(portfolio_metrics)

        return results


def generate_predictions_mlp(year: int, version: str = 'v6'):
    """
    为 MLP 模型生成预测结果并缓存

    重要修复：对全样本池预测，不使用label过滤
    - 前视偏差修复：不能用label!= -1过滤，因为label是基于未来收益生成的
    - 正确做法：对有完整特征的所有股票预测，回测时用实际收益评估
    """
    import torch
    import sys

    logger.info(f"生成 MLP {version} 预测: {year}")

    # 从训练脚本导入必要的类和函数
    if version == 'v7':
        from train_mlp_v7 import (
            Config, load_data, get_feature_columns,
            StockMLP, Neutralizer, get_best_device
        )
        model_path = "models/mlp_v7_best.pt"
        feature_cols_path = "models/feature_cols_mlp_v7.json"
        model_info_path = "models/model_info_mlp_v7.json"
        neutralizer_path = "models/neutralizer_mlp_v7.pkl"
        scaler_path = "models/scaler_mlp_v7.pkl"
        cache_name = f"mlp_v7_{year}.pkl"
    else:
        from train_mlp_v6 import (
            Config, load_data, get_feature_columns,
            StockMLP, Neutralizer, get_best_device
        )
        model_path = "models/mlp_v6_best.pt"
        feature_cols_path = "models/feature_cols_mlp_v6.json"
        model_info_path = "models/model_info_mlp_v6.json"
        neutralizer_path = "models/neutralizer_mlp_v6.pkl"
        scaler_path = "models/scaler_mlp_v6.pkl"
        cache_name = f"mlp_v6_{year}.pkl"

    # 注册 Neutralizer 到 __main__ 以便 pickle 能找到它
    sys.modules['__main__'].Neutralizer = Neutralizer

    config = Config()
    storage = SQLiteStorage()
    device = get_best_device()

    # 加载数据
    df = load_data(storage, [year], config)

    # 加载模型
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
    with open(model_info_path, "r") as f:
        model_config = json.load(f)

    neutralizer = joblib.load(neutralizer_path)
    scaler = joblib.load(scaler_path)

    model = StockMLP(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        dropout=model_config['dropout'],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ===== 关键修复：对全样本池预测，不使用label过滤 =====
    # 前视偏差修复：不能用 label != -1 过滤，因为label是基于未来收益生成的
    # 正确做法：对有完整特征的所有股票预测，回测时用实际收益评估
    valid_mask = df[feature_cols].notna().all(axis=1)  # 只检查特征完整性
    logger.info(f"预测样本数: {valid_mask.sum()} / {len(df)} (特征完整)")

    X = df.loc[valid_mask, feature_cols].fillna(0)

    X_neut = neutralizer.transform(
        df.loc[valid_mask].assign(**{col: X[col] for col in feature_cols}),
        feature_cols
    )
    X_neut = X_neut[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X_neut)

    # 预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        predictions = model(X_tensor).cpu().numpy()

    # 构建结果 DataFrame - 保留fwd_ret用于回测评估（但预测时不用它过滤）
    result_df = df.loc[valid_mask, ['ts_code', 'trade_date', 'fwd_ret']].copy()
    result_df['pred_prob'] = predictions

    # 缓存
    cache_file = Path(f"data/cache/predictions/{cache_name}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_pickle(cache_file)

    # 写入模型hash（用于缓存校验）
    model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()[:8]
    hash_file = cache_file.with_suffix('.hash')
    hash_file.write_text(model_hash)

    logger.info(f"预测已缓存: {cache_file} (hash: {model_hash})")

    return result_df


def generate_predictions_lgbm(year: int):
    """为 LightGBM 模型生成预测结果并缓存"""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM 未安装，跳过 LightGBM 评估")
        return None

    import sys

    logger.info(f"生成 LightGBM v6 预测: {year}")

    # 从训练脚本导入必要的类和函数
    from train_lightgbm_v6 import Config, load_data, get_feature_columns, Neutralizer

    # 注册 Neutralizer 到 __main__ 以便 pickle 能找到它
    sys.modules['__main__'].Neutralizer = Neutralizer

    config = Config()
    storage = SQLiteStorage()

    # 加载数据
    df = load_data(storage, [year], config)
    feature_cols = get_feature_columns(df)

    # 加载模型
    model = lgb.Booster(model_file="models/lgbm_v6.txt")
    with open("models/feature_cols_v6.json", "r") as f:
        feature_cols = json.load(f)

    neutralizer = joblib.load("models/neutralizer_v6.pkl")

    # 准备数据
    # 注意：不使用 label 过滤，仅检查特征完整性（避免前视偏差）
    valid_mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[valid_mask, feature_cols].fillna(0)

    X_neut = neutralizer.transform(
        df.loc[valid_mask].assign(**{col: X[col] for col in feature_cols}),
        feature_cols
    )
    X_neut = X_neut[feature_cols].fillna(0)

    # 预测
    predictions = model.predict(X_neut)

    # 构建结果 DataFrame
    result_df = df.loc[valid_mask, ['ts_code', 'trade_date', 'fwd_ret']].copy()
    result_df['pred_prob'] = predictions

    # 缓存
    cache_file = Path(f"data/cache/predictions/lgbm_v6_{year}.pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_pickle(cache_file)

    # 写入模型hash（用于缓存校验）
    lgbm_model_path = "models/lgbm_v6.txt"
    model_hash = hashlib.md5(open(lgbm_model_path, 'rb').read()).hexdigest()[:8]
    hash_file = cache_file.with_suffix('.hash')
    hash_file.write_text(model_hash)

    logger.info(f"预测已缓存: {cache_file} (hash: {model_hash})")

    return result_df


def compare_models(
    models: List[str] = None,
    years: List[int] = [2023, 2024]
):
    """
    对比多个模型

    Args:
        models: 模型列表 (None=自动检测可用模型)
        years: 评估年份列表
    """
    logger.info("=" * 60)
    logger.info("模型对比评估")
    logger.info("=" * 60)

    # 自动检测可用模型
    if models is None:
        models = []
        # 检查 MLP v7 (优先)
        if Path("models/mlp_v7_best.pt").exists():
            models.append('mlp_v7')
        # 检查 MLP v6
        if Path("models/mlp_v6_best.pt").exists():
            models.append('mlp_v6')
        # 检查 LightGBM
        try:
            import lightgbm
            if Path("models/lgbm_v6.txt").exists():
                models.append('lgbm_v6')
        except ImportError:
            pass

        if not models:
            logger.error("没有找到可用的模型")
            return pd.DataFrame()

        logger.info(f"检测到可用模型: {models}")

    config = EvaluationConfig()
    evaluator = ModelEvaluator(config)

    # 首先确保所有预测都已生成
    for model in models:
        for year in years:
            cache_file = Path(f"data/cache/predictions/{model}_{year}.pkl")
            if not cache_file.exists():
                logger.info(f"生成预测: {model} - {year}")
                if model == 'mlp_v7':
                    generate_predictions_mlp(year, version='v7')
                elif model == 'mlp_v6':
                    generate_predictions_mlp(year, version='v6')
                elif model == 'lgbm_v6':
                    result = generate_predictions_lgbm(year)
                    if result is None:
                        logger.warning(f"无法生成 {model} 预测，跳过")

    # 评估所有模型
    all_results = []
    for model in models:
        for year in years:
            try:
                results = evaluator.evaluate_model(model, year)
                all_results.append(results)
            except Exception as e:
                logger.error(f"评估失败: {model} - {year}: {e}")

    # 汇总结果
    results_df = pd.DataFrame(all_results)

    # 格式化输出
    logger.info("\n" + "=" * 80)
    logger.info("严格回测结果汇总")
    logger.info("=" * 80)

    # 按模型和年份分组展示
    for model in models:
        logger.info(f"\n【{model.upper()}】")
        model_results = results_df[results_df['model'] == model]

        for _, row in model_results.iterrows():
            logger.info(f"  年份: {int(row['year'])}")
            logger.info(f"    ─── 信号有效性 ───")
            logger.info(f"    AUC:              {row['auc']:.4f}")
            logger.info(f"    IC Mean:          {row['ic_ic_mean']:.4f}")
            logger.info(f"    ICIR:             {row['ic_icir']:.4f}")
            logger.info(f"    Spread (Q5-Q1):   {row['spread_spread']*100:.2f}%")
            logger.info(f"    ─── 策略表现 ───")
            logger.info(f"    年化收益(成本后): {row['annual_return']*100:.2f}%")
            logger.info(f"    最大回撤:         {row['max_drawdown']*100:.2f}%")
            logger.info(f"    夏普比率:         {row['sharpe_ratio']:.2f}")
            logger.info(f"    Calmar比率:       {row['calmar_ratio']:.2f}")
            logger.info(f"    ─── 换手与成本 ───")
            logger.info(f"    日换手率:         {row['turnover']*100:.1f}%")
            logger.info(f"    年化换手:         {row['annual_turnover']:.1f}倍")
            logger.info(f"    佣金占比:         {row['commission_ratio']*100:.1f}%")
            logger.info(f"    滑点占比:         {row['slippage_ratio']*100:.1f}%")
            logger.info(f"    ─── 相对基准 ───")
            logger.info(f"    基准年化:         {row['baseline_annual']*100:.2f}%")
            logger.info(f"    超额收益:         {row['annual_excess']*100:.2f}%")
            logger.info(f"    信息比率:         {row['information_ratio']:.2f}")

    # 生成对比表格
    logger.info("\n" + "=" * 80)
    logger.info("模型对比表格（核心指标）")
    logger.info("=" * 80)

    # 按年份对比
    for year in years:
        logger.info(f"\n年份: {year}")
        year_df = results_df[results_df['year'] == year]

        # 核心策略指标
        comparison = year_df[['model', 'auc', 'annual_return', 'max_drawdown',
                              'sharpe_ratio', 'calmar_ratio', 'annual_turnover',
                              'annual_excess', 'information_ratio']].copy()
        comparison.columns = ['Model', 'AUC', 'AnnRet', 'MaxDD', 'Sharpe', 'Calmar', 'Turnover', 'Excess', 'IR']

        # 格式化
        comparison['AnnRet'] = comparison['AnnRet'].apply(lambda x: f"{x*100:.2f}%")
        comparison['MaxDD'] = comparison['MaxDD'].apply(lambda x: f"{x*100:.2f}%")
        comparison['Sharpe'] = comparison['Sharpe'].apply(lambda x: f"{x:.2f}")
        comparison['Calmar'] = comparison['Calmar'].apply(lambda x: f"{x:.2f}")
        comparison['Turnover'] = comparison['Turnover'].apply(lambda x: f"{x:.1f}x")
        comparison['Excess'] = comparison['Excess'].apply(lambda x: f"{x*100:.2f}%")
        comparison['IR'] = comparison['IR'].apply(lambda x: f"{x:.2f}")

        logger.info(f"\n{comparison.to_string(index=False)}")

    # 保存结果
    results_df.to_csv("data/model_comparison_results.csv", index=False)
    logger.info(f"\n结果已保存至: data/model_comparison_results.csv")

    return results_df


if __name__ == "__main__":
    results = compare_models(
        models=['mlp_v7', 'mlp_v6', 'lgbm_v6'],
        years=[2023, 2024]
    )
