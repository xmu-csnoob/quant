#!/usr/bin/env python3
"""
PyTorch MLP量化选股模型 V6 - 深度学习版

基于 LightGBM v6 改写，核心改进：
1. 截面排名标签 - 预测相对排名而非绝对方向
2. 扩展因子库 - 量价+截面+资金流+技术指标
3. 行业市值中性化 - 剥离Beta，保留纯Alpha
4. PyTorch MLP模型 - 深度学习，支持MPS加速
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import warnings
import hashlib
import json
import joblib

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# 特征缓存目录
CACHE_DIR = Path("data/cache/features")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 缓存格式: 使用 pickle 避免 pyarrow 依赖
CACHE_FORMAT = "pkl"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage.sqlite_storage import SQLiteStorage
from src.models.lstm_model import get_best_device


# ==================== 配置 ====================

class Config:
    """训练配置 - V7 优化版"""
    # 数据范围
    train_years = [2021, 2022]
    val_year = 2023
    test_year = 2024

    # 标签设置 - 更严格
    prediction_period = 20  # 预测未来20天
    label_type = "rank"  # "rank" 截面排名, "direction" 方向
    top_quantile = 0.2   # 前20%为正样本（更严格）
    bottom_quantile = 0.2  # 后20%为负样本

    # 中性化
    neutralize_industry = True
    neutralize_size = True

    # 模型参数 - 简化+正则化
    hidden_sizes = [128, 64, 32]  # 减少层数
    dropout = 0.5  # 增加dropout
    use_batch_norm = True

    # 训练参数 - 加强正则化
    batch_size = 4096
    learning_rate = 5e-4  # 降低学习率
    weight_decay = 1e-3   # 增加L2正则化
    epochs = 100
    early_stopping_patience = 15
    num_workers = 4

    # 设备
    device = "auto"


# ==================== 因子计算 ====================

class FactorCalculator:
    """因子计算器 - 扩展因子库"""

    def __init__(self, storage: SQLiteStorage):
        self.storage = storage

    def compute_price_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价因子"""
        df = df.copy()
        df = df.sort_values(['ts_code', 'trade_date'])

        g = df.groupby('ts_code')

        # ===== 收益率因子 =====
        df['ret_1d'] = g['close'].pct_change()
        df['ret_5d'] = g['close'].pct_change(5)
        df['ret_10d'] = g['close'].pct_change(10)
        df['ret_20d'] = g['close'].pct_change(20)

        # ===== 均线因子 =====
        df['ma_5'] = g['close'].transform(lambda x: x.rolling(5).mean())
        df['ma_10'] = g['close'].transform(lambda x: x.rolling(10).mean())
        df['ma_20'] = g['close'].transform(lambda x: x.rolling(20).mean())
        df['ma_60'] = g['close'].transform(lambda x: x.rolling(60).mean())

        # MA偏离度
        df['bias_ma5'] = (df['close'] - df['ma_5']) / df['ma_5']
        df['bias_ma10'] = (df['close'] - df['ma_10']) / df['ma_10']
        df['bias_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']

        # MA斜率 (趋势强度)
        df['ma5_slope'] = g['ma_5'].transform(lambda x: x.pct_change(5))
        df['ma20_slope'] = g['ma_20'].transform(lambda x: x.pct_change(5))

        # ===== 波动率因子 =====
        df['volatility_5d'] = g['ret_1d'].transform(lambda x: x.rolling(5).std())
        df['volatility_10d'] = g['ret_1d'].transform(lambda x: x.rolling(10).std())
        df['volatility_20d'] = g['ret_1d'].transform(lambda x: x.rolling(20).std())

        # ATR
        df['high_low'] = df['high'] - df['low']
        # 修复：使用分组shift避免跨股票污染
        df['high_close'] = abs(df['high'] - g['close'].shift(1).values)
        df['low_close'] = abs(df['low'] - g['close'].shift(1).values)
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = g['tr'].transform(lambda x: x.rolling(14).mean())
        df['atr_ratio'] = df['atr_14'] / df['close']

        # ===== 成交量因子 =====
        df['vol_ma_5'] = g['vol'].transform(lambda x: x.rolling(5).mean())
        df['vol_ma_20'] = g['vol'].transform(lambda x: x.rolling(20).mean())
        df['vol_ratio_5'] = df['vol'] / df['vol_ma_5']
        df['vol_ratio_20'] = df['vol'] / df['vol_ma_20']

        # 成交量变化率
        df['vol_chg_1d'] = g['vol'].pct_change()
        df['vol_chg_5d'] = g['vol'].pct_change(5)

        # ===== 动量因子 =====
        df['momentum_5d'] = df['ret_5d']
        df['momentum_10d'] = df['ret_10d']
        df['momentum_20d'] = df['ret_20d']

        # 相对强度
        df['price_rank_20d'] = g['close'].transform(
            lambda x: x.rolling(20).rank(pct=True)
        )

        # ===== 反转因子 =====
        df['reversal_5d'] = -df['ret_5d']
        df['reversal_10d'] = -df['ret_10d']

        # ===== RSI =====
        delta = g['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.groupby(df['ts_code']).transform(lambda x: x.rolling(14).mean())
        avg_loss = loss.groupby(df['ts_code']).transform(lambda x: x.rolling(14).mean())
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)

        # ===== MACD =====
        ema_12 = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema_26 = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = g['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 修复：使用分组shift避免跨股票污染
        macd_prev = g['macd'].shift(1).values
        signal_prev = g['macd_signal'].shift(1).values
        df['macd_golden_cross'] = ((df['macd'] > df['macd_signal']) &
                                   (macd_prev <= signal_prev)).astype(int)

        # ===== 布林带 =====
        df['boll_mid'] = g['close'].transform(lambda x: x.rolling(20).mean())
        df['boll_std'] = g['close'].transform(lambda x: x.rolling(20).std())
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        df['boll_position'] = (df['close'] - df['boll_mid']) / (2 * df['boll_std'] + 1e-10)
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']

        # ===== KDJ =====
        low_9 = g['low'].transform(lambda x: x.rolling(9).min())
        high_9 = g['high'].transform(lambda x: x.rolling(9).max())
        df['kdj_rsv'] = (df['close'] - low_9) / (high_9 - low_9 + 1e-10) * 100
        df['kdj_k'] = g['kdj_rsv'].transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())
        df['kdj_d'] = g['kdj_k'].transform(lambda x: x.ewm(alpha=1/3, adjust=False).mean())
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        # ===== 价格形态因子 =====
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['is_up'] = (df['close'] > df['open']).astype(int)

        # 连续上涨天数
        df['up_days'] = g['is_up'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )

        # ===== 换手率因子 =====
        df['amount_ma_5'] = g['amount'].transform(lambda x: x.rolling(5).mean())
        df['amount_ratio'] = df['amount'] / (df['amount_ma_5'] + 1e-10)

        # 清理临时列
        df = df.drop(columns=['high_low', 'high_close', 'low_close', 'tr', 'boll_mid',
                               'boll_std', 'boll_upper', 'boll_lower'], errors='ignore')

        return df

    def load_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载截面特征"""
        try:
            cs_df = pd.read_sql_query(
                "SELECT * FROM cross_sectional_features",
                self.storage.conn
            )
            if cs_df.empty:
                logger.warning("截面特征表为空")
                return df

            cs_df['trade_date'] = cs_df['trade_date'].astype(str)
            df = df.merge(cs_df, on=['ts_code', 'trade_date'], how='left')
            logger.info(f"加载截面特征: {len(cs_df.columns) - 2} 个")
        except Exception as e:
            logger.warning(f"加载截面特征失败: {e}")

        return df

    def load_basic_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载基本面因子"""
        try:
            bf_df = pd.read_sql_query(
                "SELECT ts_code, trade_date, f_amount_rank, f_log_size, f_liquidity, "
                "f_volatility, f_illiq, f_momentum_20, f_momentum_60, f_price_pos, f_turnover "
                "FROM basic_factors",
                self.storage.conn
            )
            if bf_df.empty:
                logger.warning("基本面因子表为空")
                return df

            bf_df['trade_date'] = bf_df['trade_date'].astype(str)
            df = df.merge(bf_df, on=['ts_code', 'trade_date'], how='left')
            logger.info(f"加载基本面因子: {len(bf_df.columns) - 2} 个")
        except Exception as e:
            logger.warning(f"加载基本面因子失败: {e}")

        return df

    def load_fund_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载资金流向数据"""
        try:
            ff_df = pd.read_sql_query(
                "SELECT ts_code, trade_date, main_net_ratio, super_net_ratio, "
                "large_net_ratio, medium_net_ratio, small_net_ratio "
                "FROM fund_flow",
                self.storage.conn
            )
            if ff_df.empty:
                logger.warning("资金流向表为空")
                return df

            ff_df['trade_date'] = ff_df['trade_date'].astype(str)
            df = df.merge(ff_df, on=['ts_code', 'trade_date'], how='left')
            logger.info(f"加载资金流向: {len(ff_df.columns) - 2} 个")
        except Exception as e:
            logger.warning(f"加载资金流向失败: {e}")

        return df

    def load_proxy_financial_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载代理财务因子"""
        try:
            pf_df = pd.read_sql_query(
                "SELECT ts_code, trade_date, f_value_proxy, f_quality_proxy, "
                "f_profit_proxy, f_growth_proxy, f_leverage_proxy "
                "FROM proxy_financial_factors",
                self.storage.conn
            )
            if pf_df.empty:
                logger.warning("代理财务因子表为空")
                return df

            pf_df['trade_date'] = pf_df['trade_date'].astype(str)
            df = df.merge(pf_df, on=['ts_code', 'trade_date'], how='left')
            logger.info(f"加载代理财务因子: {len(pf_df.columns) - 2} 个")
        except Exception as e:
            logger.warning(f"加载代理财务因子失败: {e}")

        return df

    def load_extended_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载扩展因子"""
        try:
            ext_df = pd.read_sql_query(
                "SELECT * FROM extended_factors",
                self.storage.conn
            )
            if ext_df.empty:
                logger.warning("扩展因子表为空")
                return df

            ext_df['trade_date'] = ext_df['trade_date'].astype(str)
            df = df.merge(ext_df, on=['ts_code', 'trade_date'], how='left')
            logger.info(f"加载扩展因子: {len(ext_df.columns) - 2} 个")
        except Exception as e:
            logger.warning(f"加载扩展因子失败: {e}")

        return df

    def load_industry_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """加载行业分类"""
        try:
            ind_df = pd.read_sql_query(
                "SELECT ts_code, industry FROM industry_classification",
                self.storage.conn
            )
            if not ind_df.empty:
                df = df.merge(ind_df, on='ts_code', how='left')
                logger.info(f"加载行业分类: {ind_df['industry'].nunique()} 个行业")
        except Exception as e:
            logger.warning(f"加载行业分类失败: {e}")
            df['industry'] = 'Unknown'

        return df


# ==================== 中性化处理 ====================

class Neutralizer:
    """
    因子中性化处理

    修复时间泄漏：改为按交易日截面处理
    - 每个交易日独立进行中性化回归
    - 避免使用未来日期的行业分布信息
    """

    def __init__(self):
        self.industry_cols = None  # 所有可能的行业
        self.fitted_industries = set()  # 训练集见过的行业

    def fit(self, df: pd.DataFrame, feature_cols: list):
        """拟合中性化模型 - 记录所有行业类别"""
        if 'industry' in df.columns:
            # 过滤掉 NaN 值
            all_industries = df['industry'].dropna().unique()
            self.industry_cols = [f'ind_{ind}' for ind in sorted(all_industries)]
            self.fitted_industries = set(all_industries)
            logger.info(f"中性化: {len(self.industry_cols)} 个行业哑变量")

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        应用中性化 - 按交易日截面处理

        关键修复：每个交易日独立回归，        """
        df = df.copy()

        # 按交易日分组处理
        for trade_date in df['trade_date'].unique():
            date_mask = df['trade_date'] == trade_date
            date_df = df.loc[date_mask].copy()

            X_neutral = []

            # 行业哑变量
            if self.industry_cols is not None and 'industry' in date_df.columns:
                ind_dummies = pd.get_dummies(date_df['industry'], prefix='ind')
                # 确保列对齐
                for col in self.industry_cols:
                    if col not in ind_dummies.columns:
                        ind_dummies[col] = 0
                X_neutral.append(ind_dummies[self.industry_cols].values)

            # 对数市值
            if 'amount' in date_df.columns:
                log_amount = np.log1p(date_df['amount'].values.reshape(-1, 1))
                X_neutral.append(log_amount)

            if not X_neutral:
                continue

            X = np.hstack(X_neutral)

            # 对每个因子进行截面中性化
            for col in feature_cols:
                if col in date_df.columns:
                    y = date_df[col].values
                    try:
                        # 检查是否有足够样本
                        if len(y) > len(X[0]) + 5:  # 样本数 > 特征数 + 5
                            coef = np.linalg.lstsq(X, y, rcond=None)[0]
                            residual = y - X @ coef
                            df.loc[date_mask, col] = residual
                    except:
                        pass

        return df


# ==================== 标签计算 ====================

def compute_labels(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """计算截面排名标签"""
    df = df.copy()
    df = df.sort_values(['ts_code', 'trade_date'])

    df['fwd_ret'] = df.groupby('ts_code')['close'].transform(
        lambda x: x.shift(-config.prediction_period) / x - 1
    )

    if config.label_type == "rank":
        df['ret_rank'] = df.groupby('trade_date')['fwd_ret'].rank(pct=True)
        df['label'] = -1
        df.loc[df['ret_rank'] > (1 - config.top_quantile), 'label'] = 1
        df.loc[df['ret_rank'] < config.bottom_quantile, 'label'] = 0

        logger.info(f"标签分布: 0={sum(df['label']==0)}, 1={sum(df['label']==1)}, 丢弃={sum(df['label']==-1)}")
    else:
        df['label'] = (df['fwd_ret'] > 0).astype(int)

    return df


# ==================== PyTorch 模型 ====================

class StockMLP(nn.Module):
    """
    股票选股MLP模型

    架构：
        输入层 (n_features)
            ↓
        [BatchNorm1d]
            ↓
        [FC Layer 1] n_features -> hidden_sizes[0], ReLU, BatchNorm, Dropout
            ↓
        [FC Layer 2] hidden[0] -> hidden[1], ReLU, BatchNorm, Dropout
            ↓
        ...
            ↓
        [Output] hidden[-1] -> 1, Sigmoid
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [256, 128, 64, 32],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        layers = []

        # 输入批归一化
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_size))

        # 隐藏层
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

        logger.info(f"StockMLP initialized: input={input_size}, hidden={hidden_sizes}, dropout={dropout}")

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (batch, features)

        Returns:
            预测概率 (batch, 1)
        """
        return self.model(x).squeeze(-1)

    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StockDataset(Dataset):
    """股票数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 主训练流程 ====================

def load_data(storage: SQLiteStorage, years: list, config: Config):
    """加载并处理数据（带缓存）"""
    # 缓存键必须包含所有影响标签计算的参数
    cache_key = hashlib.md5(
        f"{years}_{config.prediction_period}_{config.label_type}_"
        f"top{config.top_quantile}_btm{config.bottom_quantile}".encode()
    ).hexdigest()
    cache_file = CACHE_DIR / f"features_{cache_key}.{CACHE_FORMAT}"

    if cache_file.exists():
        logger.info(f"从缓存加载特征: {cache_file}")
        df = pd.read_pickle(cache_file)
        logger.info(f"缓存数据: {len(df)} 行")
        return df

    logger.info(f"加载 {years} 年数据...")

    year_patterns = " OR ".join([f"trade_date LIKE '{y}%'" for y in years])
    query = f"""
    SELECT ts_code, trade_date, open, high, low, close, vol, amount
    FROM daily_prices
    WHERE {year_patterns}
    ORDER BY ts_code, trade_date
    """

    df = pd.read_sql_query(query, storage.conn)
    logger.info(f"原始数据: {len(df)} 行, {df['ts_code'].nunique()} 只股票")

    calculator = FactorCalculator(storage)
    df = calculator.compute_price_factors(df)
    df = calculator.load_cross_sectional_features(df)
    df = calculator.load_fund_flow(df)
    df = calculator.load_basic_factors(df)
    df = calculator.load_proxy_financial_factors(df)
    df = calculator.load_extended_factors(df)
    df = calculator.load_industry_info(df)

    df = compute_labels(df, config)

    logger.info(f"处理后数据: {len(df)} 行")

    df.to_pickle(cache_file)
    logger.info(f"特征已缓存: {cache_file}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列"""
    exclude_cols = [
        'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount',
        'fwd_ret', 'ret_rank', 'label', 'industry',
        'ma_5', 'ma_10', 'ma_20', 'ma_60',
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('ind_')]

    # 移除常数列
    constant_cols = []
    for col in feature_cols:
        if df[col].std() < 1e-10:
            constant_cols.append(col)

    if constant_cols:
        logger.warning(f"移除常数列: {constant_cols}")
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    return feature_cols


def train_model():
    """训练 PyTorch MLP 模型"""
    import time
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PyTorch MLP 量化选股模型 V7 - 优化版")
    logger.info("=" * 60)

    config = Config()
    storage = SQLiteStorage()

    # 设备设置
    if config.device == "auto":
        device = get_best_device()
    else:
        device = config.device
    logger.info(f"使用设备: {device}")

    # 加载训练数据
    train_df = load_data(storage, config.train_years, config)
    val_df = load_data(storage, [config.val_year], config)

    # 获取特征列
    feature_cols = get_feature_columns(train_df)
    logger.info(f"特征数量: {len(feature_cols)}")

    # 准备训练数据
    train_mask = train_df['label'] != -1
    X_train = train_df.loc[train_mask, feature_cols].fillna(0)
    y_train = train_df.loc[train_mask, 'label'].values

    val_mask = val_df['label'] != -1
    X_val = val_df.loc[val_mask, feature_cols].fillna(0)
    y_val = val_df.loc[val_mask, 'label'].values

    logger.info(f"训练集: {len(X_train)} 样本")
    logger.info(f"验证集: {len(X_val)} 样本")

    # 中性化处理
    logger.info("应用中性化处理...")
    neutralizer = Neutralizer()
    neutralizer.fit(train_df.loc[train_mask], feature_cols)

    X_train_neut = neutralizer.transform(
        train_df.loc[train_mask].assign(**{col: X_train[col] for col in feature_cols}),
        feature_cols
    )
    X_train_neut = X_train_neut[feature_cols].fillna(0).values

    X_val_neut = neutralizer.transform(
        val_df.loc[val_mask].assign(**{col: X_val[col] for col in feature_cols}),
        feature_cols
    )
    X_val_neut = X_val_neut[feature_cols].fillna(0).values

    # 标准化
    logger.info("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_neut)
    X_val_scaled = scaler.transform(X_val_neut)

    # 创建数据集
    train_dataset = StockDataset(X_train_scaled, y_train)
    val_dataset = StockDataset(X_val_scaled, y_val)

    # 数据加载器
    use_cuda_pin = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_cuda_pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_cuda_pin,
    )

    # 创建模型
    model = StockMLP(
        input_size=len(feature_cols),
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
        use_batch_norm=config.use_batch_norm,
    ).to(device)

    logger.info(f"模型参数: {model.count_parameters():,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 损失函数
    criterion = nn.BCELoss()

    # 训练循环
    best_val_loss = float('inf')
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        # 训练
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        # 计算指标
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(val_labels, val_preds)

        # 学习率调度
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存最佳模型
            torch.save(model.state_dict(), "models/mlp_v7_best.pt")
            logger.info(f"  -> 保存最佳模型 (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load("models/mlp_v7_best.pt", map_location=device))
    model.eval()

    # 最终评估
    with torch.no_grad():
        train_preds = []
        for batch_X, _ in DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            train_preds.extend(outputs.cpu().numpy())

        val_preds = []
        for batch_X, _ in DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            val_preds.extend(outputs.cpu().numpy())

    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)

    logger.info(f"\n{'='*40}")
    logger.info(f"训练结果")
    logger.info(f"{'='*40}")
    logger.info(f"Train AUC: {train_auc:.4f}")
    logger.info(f"Val AUC: {val_auc:.4f}")

    # 保存模型和配置 (使用 _mlp 后缀避免与 LightGBM 冲突)
    logger.info(f"\n模型保存至: models/mlp_v7_best.pt")

    with open("models/feature_cols_mlp_v7.json", "w") as f:
        json.dump(feature_cols, f)

    joblib.dump(neutralizer, "models/neutralizer_mlp_v7.pkl")
    joblib.dump(scaler, "models/scaler_mlp_v7.pkl")

    training_time = time.time() - start_time

    with open("models/model_info_mlp_v7.json", "w") as f:
        json.dump({
            'model_type': 'MLP',
            'version': 'v7',
            'hidden_sizes': config.hidden_sizes,
            'dropout': config.dropout,
            'input_size': len(feature_cols),
            'train_years': config.train_years,
            'val_year': config.val_year,
            'test_year': config.test_year,
            'prediction_period': config.prediction_period,
            'label_type': config.label_type,
            'top_quantile': config.top_quantile,
            'bottom_quantile': config.bottom_quantile,
            # 性能指标
            'train_auc': train_auc,
            'val_auc': val_auc,
            'training_time_seconds': round(training_time, 2),
            'training_time_formatted': f"{int(training_time // 60)}m {int(training_time % 60)}s",
            'best_epoch': best_epoch,
            # 数据版本
            'created_at': pd.Timestamp.now().isoformat(),
        }, f, indent=2)

    return {
        'train_auc': train_auc,
        'val_auc': val_auc,
    }


def validate_signal_quality():
    """
    信号分层有效性验证（非真实交易回测）

    警告：此函数仅验证预测信号的分层有效性，不包含：
    - T+1 约束（A股当日买入次日才能卖出）
    - 涨跌停限制（涨停无法买入，跌停无法卖出）
    - 停牌处理
    - 成交量约束
    - 交易成本与滑点

    真实策略回测请使用 apps/evaluate_model_comparison.py
    """
    logger.info("\n" + "=" * 60)
    logger.info("信号分层有效性验证（非真实交易回测）")
    logger.info("=" * 60)

    config = Config()
    storage = SQLiteStorage()

    # 设备
    device = get_best_device()
    logger.info(f"使用设备: {device}")

    # 加载测试数据
    test_df = load_data(storage, [config.test_year], config)

    # 加载模型
    with open("models/feature_cols_mlp_v7.json", "r") as f:
        feature_cols = json.load(f)

    with open("models/model_info_mlp_v7.json", "r") as f:
        model_config = json.load(f)

    neutralizer = joblib.load("models/neutralizer_mlp_v7.pkl")
    scaler = joblib.load("models/scaler_mlp_v7.pkl")

    model = StockMLP(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        dropout=model_config['dropout'],
    ).to(device)
    model.load_state_dict(torch.load("models/mlp_v7_best.pt", map_location=device))
    model.eval()

    # 准备测试数据
    test_mask = test_df['label'] != -1
    X_test = test_df.loc[test_mask, feature_cols].fillna(0)

    X_test_neut = neutralizer.transform(
        test_df.loc[test_mask].assign(**{col: X_test[col] for col in feature_cols}),
        feature_cols
    )
    X_test_neut = X_test_neut[feature_cols].fillna(0).values
    X_test_scaled = scaler.transform(X_test_neut)

    y_test = test_df.loc[test_mask, 'label'].values

    # 预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        test_pred = model(X_tensor).cpu().numpy()

    test_auc = roc_auc_score(y_test, test_pred)
    logger.info(f"Test AUC (2024): {test_auc:.4f}")

    # 分组回测
    test_df.loc[test_mask, 'pred_prob'] = test_pred
    test_df.loc[test_mask, 'pred_group'] = pd.qcut(test_pred, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # 各组收益
    group_returns = test_df.loc[test_mask].groupby('pred_group')['fwd_ret'].mean()
    logger.info(f"\n分组收益:")
    logger.info(f"\n{group_returns.to_string()}")

    # Top vs Bottom
    top_return = group_returns.get('Q5', 0)
    bottom_return = group_returns.get('Q1', 0)
    spread = top_return - bottom_return

    logger.info(f"\nTop(Q5)收益: {top_return*100:.2f}%")
    logger.info(f"Bottom(Q1)收益: {bottom_return*100:.2f}%")
    logger.info(f"Spread: {spread*100:.2f}%")

    return {
        'test_auc': test_auc,
        'group_returns': group_returns.to_dict(),
        'spread': spread
    }


if __name__ == "__main__":
    # 训练
    train_results = train_model()

    # 信号有效性验证（非真实交易回测）
    signal_results = validate_signal_quality()

    logger.info("\n" + "=" * 60)
    logger.info("最终结果汇总")
    logger.info("=" * 60)
    logger.info(f"Train AUC: {train_results['train_auc']:.4f}")
    logger.info(f"Val AUC: {train_results['val_auc']:.4f}")
    logger.info(f"Test AUC: {signal_results['test_auc']:.4f}")
    logger.info(f"Spread (Q5-Q1): {signal_results['spread']*100:.2f}%")
    logger.info("\n注意: 以上为信号分层验证，非真实交易回测。")
    logger.info("完整回测请运行: python apps/evaluate_model_comparison.py")
