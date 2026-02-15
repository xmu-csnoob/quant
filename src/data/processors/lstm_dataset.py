"""
LSTM时序数据集

将股票数据转换为适合LSTM训练的时序样本：
- 输入: (seq_len, features) 历史序列
- 输出: 未来涨跌标签 (0/1)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Callable
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from loguru import logger


class LSTMDataset(Dataset):
    """
    LSTM时序数据集

    将DataFrame转换为(seq_len, features)的时序样本
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 20,
    ):
        """
        初始化数据集

        Args:
            data: 特征数据 (n_samples, n_features)
            labels: 标签 (n_samples,)
            seq_len: 序列长度
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len

        # 验证数据
        assert len(data) == len(labels), "数据和标签数量不匹配"
        assert len(data) >= seq_len, f"数据长度 {len(data)} 小于序列长度 {seq_len}"

    def __len__(self) -> int:
        """返回样本数量"""
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            x: 特征序列 (seq_len, n_features)
            y: 标签 (1,)
        """
        # 获取seq_len天的历史数据
        x = self.data[idx:idx + self.seq_len]
        # 标签是seq_len后的涨跌
        y = self.labels[idx + self.seq_len].unsqueeze(0)
        return x, y


class SequenceDataset(Dataset):
    """
    通用时序数据集

    支持预构造好的序列数据
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
    ):
        """
        初始化

        Args:
            sequences: 序列数据 (n_samples, seq_len, n_features)
            labels: 标签 (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class LSTMDatasetBuilder:
    """
    LSTM数据集构建器

    从原始股票数据构建训练数据集
    """

    def __init__(
        self,
        seq_len: int = 20,
        prediction_period: int = 5,
        buy_threshold: float = 0.0,
        feature_cols: Optional[List[str]] = None,
    ):
        """
        初始化构建器

        Args:
            seq_len: 序列长度（历史天数）
            prediction_period: 预测周期（未来天数）
            buy_threshold: 买入阈值（涨跌幅超过此值为涨）
            feature_cols: 特征列名列表
        """
        self.seq_len = seq_len
        self.prediction_period = prediction_period
        self.buy_threshold = buy_threshold
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_extractor: Optional[Callable] = None,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签

        Args:
            df: 原始OHLCV数据
            feature_extractor: 特征提取器函数
            fit_scaler: 是否拟合scaler

        Returns:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
        """
        # 提取特征
        if feature_extractor is not None:
            df = feature_extractor(df)

        # 获取特征列
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c.startswith("f_")]

        # 过滤有效数据
        df_valid = df.dropna(subset=self.feature_cols).copy()

        if len(df_valid) == 0:
            logger.warning("没有有效数据")
            return np.array([]), np.array([])

        # 提取特征
        features = df_valid[self.feature_cols].values

        # 处理NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        # 创建标签：未来N天的累计收益
        df_valid['future_return'] = df_valid['close'].pct_change(self.prediction_period).shift(-self.prediction_period)
        labels = (df_valid['future_return'] > self.buy_threshold).astype(float).values

        # 移除末尾无效的标签
        features = features[:-self.prediction_period]
        labels = labels[:-self.prediction_period]

        logger.info(f"准备特征: {features.shape}, 标签: {labels.shape}")
        logger.info(f"正样本比例: {labels.mean():.2%}")

        return features, labels

    def build_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建时序序列

        Args:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)

        Returns:
            sequences: 序列数组 (n_sequences, seq_len, n_features)
            seq_labels: 序列标签 (n_sequences,)
        """
        n_samples = len(features) - self.seq_len
        n_features = features.shape[1]

        sequences = np.zeros((n_samples, self.seq_len, n_features), dtype=np.float32)
        seq_labels = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            sequences[i] = features[i:i + self.seq_len]
            seq_labels[i] = labels[i + self.seq_len - 1]

        logger.info(f"构建序列: {sequences.shape}, 标签: {seq_labels.shape}")
        return sequences, seq_labels

    def build_dataset(
        self,
        df: pd.DataFrame,
        feature_extractor: Optional[Callable] = None,
        fit_scaler: bool = True,
    ) -> SequenceDataset:
        """
        构建数据集

        Args:
            df: 原始数据
            feature_extractor: 特征提取器
            fit_scaler: 是否拟合scaler

        Returns:
            SequenceDataset
        """
        features, labels = self.prepare_features(df, feature_extractor, fit_scaler)

        if len(features) == 0:
            raise ValueError("没有有效数据")

        sequences, seq_labels = self.build_sequences(features, labels)

        return SequenceDataset(sequences, seq_labels)

    def save_scaler(self, path: str):
        """保存scaler"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler保存: {path}")

    def load_scaler(self, path: str):
        """加载scaler"""
        self.scaler = joblib.load(path)
        logger.info(f"Scaler加载: {path}")


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 512,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    创建数据加载器

    Args:
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集
        batch_size: 批大小
        num_workers: 工作进程数
        pin_memory: 是否锁页内存

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = "trade_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按日期划分数据集

    Args:
        df: 原始数据
        train_end: 训练集结束日期 (如 "2022-12-31")
        val_end: 验证集结束日期 (如 "2023-06-30")
        date_col: 日期列名

    Returns:
        train_df, val_df, test_df
    """
    # 确保日期格式
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    train_df = df[df[date_col] <= train_end].copy()
    val_df = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test_df = df[df[date_col] > val_end].copy()

    logger.info(f"数据划分: 训练={len(train_df)}, 验证={len(val_df)}, 测试={len(test_df)}")

    return train_df, val_df, test_df
