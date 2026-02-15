#!/usr/bin/env python3
"""
LSTM模型训练脚本

完整的训练流水线：
1. 从数据库加载股票数据
2. 提取特征
3. 划分训练/验证/测试集
4. 训练LSTM模型
5. 评估并保存模型

使用方法:
    # 快速测试（100只股票）
    python scripts/train_lstm_model.py --max-stocks 100

    # 完整训练
    python scripts/train_lstm_model.py

    # 指定GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train_lstm_model.py
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import StockLSTMClassifier, create_lstm_model
from src.data.processors.lstm_dataset import (
    SequenceDataset,
    LSTMDatasetBuilder,
    create_dataloaders,
)
from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor


class LSTMTrainer:
    """LSTM模型训练器"""

    def __init__(
        self,
        model: StockLSTMClassifier,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
    ):
        """
        初始化训练器

        Args:
            model: LSTM模型
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            grad_clip: 梯度裁剪
        """
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip

        # 损失函数（带类别权重处理不平衡）
        self.criterion = nn.BCELoss()

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 初始周期
            T_mult=2,  # 周期倍增因子
        )

        # 记录最佳模型
        self.best_val_auc = 0.0
        self.best_model_state = None
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch

        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs, _ = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # 记录
            total_loss += loss.item() * batch_x.size(0)
            preds = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
            labels = batch_y.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float, float]:
        """
        验证

        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch

        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
            auc: AUC
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs, _ = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item() * batch_x.size(0)
                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                labels = batch_y.cpu().numpy().flatten()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        # 计算AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return avg_loss, accuracy, auc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: str = "models",
    ) -> dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            num_epochs: 最大epoch数
            patience: 早停耐心值
            checkpoint_dir: 检查点目录

        Returns:
            训练历史
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
        }

        logger.info(f"开始训练，最大epochs={num_epochs}, 早停patience={patience}")

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # 验证
            val_loss, val_acc, val_auc = self.validate(val_loader, epoch)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)

            # 学习率调度
            self.scheduler.step()

            # 打印摘要
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}"
            )

            # 保存最佳模型
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0

                # 保存检查点
                torch.save(
                    self.best_model_state,
                    Path(checkpoint_dir) / "lstm_best.pt",
                )
                logger.info(f"保存最佳模型 (AUC={val_auc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"早停：验证AUC连续{patience}个epoch未提升")
                    break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> dict:
        """
        评估测试集

        Args:
            test_loader: 测试数据加载器

        Returns:
            评估指标
        """
        self.model.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
                batch_x = batch_x.to(self.device)
                outputs, _ = self.model(batch_x)

                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                labels = batch_y.numpy().flatten()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }

        try:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metrics["auc"] = 0.5

        return metrics


def load_stock_data(
    storage: SQLiteStorage,
    max_stocks: Optional[int] = None,
    min_data_days: int = 100,
) -> list[pd.DataFrame]:
    """
    从数据库加载股票数据

    Args:
        storage: 数据存储
        max_stocks: 最大股票数量（用于测试）
        min_data_days: 最小数据天数

    Returns:
        股票数据列表
    """
    logger.info("从数据库加载股票数据...")

    # 获取所有股票代码
    all_stocks = storage.get_all_stocks()
    if max_stocks:
        all_stocks = all_stocks[:max_stocks]

    logger.info(f"共{len(all_stocks)}只股票")

    # 加载数据
    stock_data_list = []
    for code in tqdm(all_stocks, desc="加载数据"):
        try:
            df = storage.get_daily_prices(code)
            if df is not None and len(df) >= min_data_days:
                stock_data_list.append(df)
        except Exception as e:
            logger.warning(f"加载 {code} 失败: {e}")

    logger.info(f"成功加载 {len(stock_data_list)} 只股票数据")
    return stock_data_list


def build_datasets(
    stock_data_list: list[pd.DataFrame],
    feature_extractor: EnhancedFeatureExtractor,
    seq_len: int = 20,
    prediction_period: int = 5,
    train_end: str = "2022-12-31",
    val_end: str = "2023-06-30",
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset, LSTMDatasetBuilder]:
    """
    构建训练/验证/测试数据集

    Args:
        stock_data_list: 股票数据列表
        feature_extractor: 特征提取器
        seq_len: 序列长度
        prediction_period: 预测周期
        train_end: 训练集结束日期
        val_end: 验证集结束日期

    Returns:
        train_dataset, val_dataset, test_dataset, dataset_builder
    """
    logger.info("构建数据集...")

    dataset_builder = LSTMDatasetBuilder(
        seq_len=seq_len,
        prediction_period=prediction_period,
        buy_threshold=0.0,
    )

    train_datasets = []
    val_datasets = []
    test_datasets = []

    # 第一遍：构建训练集并拟合scaler
    logger.info("处理训练集数据...")
    for df in tqdm(stock_data_list, desc="构建训练集"):
        try:
            # 按日期划分
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            train_df = df[df['trade_date'] <= train_end].copy()

            if len(train_df) < seq_len + prediction_period + 60:
                continue

            # 提取特征
            train_df = feature_extractor.extract(train_df)

            # 构建数据集
            dataset = dataset_builder.build_dataset(
                train_df,
                fit_scaler=True,
            )
            if len(dataset) > 0:
                train_datasets.append(dataset)
        except Exception as e:
            pass

    # 合并训练集
    if not train_datasets:
        raise ValueError("没有有效的训练数据")

    train_dataset = ConcatDataset(train_datasets)
    logger.info(f"训练集样本数: {len(train_dataset)}")

    # 第二遍：构建验证集和测试集（使用已拟合的scaler）
    logger.info("处理验证/测试集数据...")
    for df in tqdm(stock_data_list, desc="构建验证/测试集"):
        try:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            train_end_dt = pd.to_datetime(train_end)
            val_end_dt = pd.to_datetime(val_end)

            train_df = df[df['trade_date'] <= train_end_dt].copy()
            val_df = df[(df['trade_date'] > train_end_dt) & (df['trade_date'] <= val_end_dt)].copy()
            test_df = df[df['trade_date'] > val_end_dt].copy()

            # 提取特征
            if len(val_df) >= seq_len + prediction_period + 10:
                val_df = feature_extractor.extract(val_df)
                val_dataset = dataset_builder.build_dataset(val_df, fit_scaler=False)
                if len(val_dataset) > 0:
                    val_datasets.append(val_dataset)

            if len(test_df) >= seq_len + prediction_period + 10:
                test_df = feature_extractor.extract(test_df)
                test_dataset = dataset_builder.build_dataset(test_df, fit_scaler=False)
                if len(test_dataset) > 0:
                    test_datasets.append(test_dataset)
        except Exception as e:
            pass

    # 合并数据集
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    test_dataset = ConcatDataset(test_datasets) if test_datasets else None

    logger.info(f"验证集样本数: {len(val_dataset) if val_dataset else 0}")
    logger.info(f"测试集样本数: {len(test_dataset) if test_dataset else 0}")

    return train_dataset, val_dataset, test_dataset, dataset_builder


def main():
    parser = argparse.ArgumentParser(description="训练LSTM模型")
    parser.add_argument("--max-stocks", type=int, default=None, help="最大股票数量（用于测试）")
    parser.add_argument("--epochs", type=int, default=50, help="最大训练epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seq-len", type=int, default=20, help="序列长度")
    parser.add_argument("--prediction-period", type=int, default=5, help="预测周期")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    parser.add_argument("--output-dir", type=str, default="models", help="输出目录")
    args = parser.parse_args()

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"使用设备: {device}")

    # 加载数据
    storage = SQLiteStorage()
    stock_data_list = load_stock_data(storage, args.max_stocks)

    if not stock_data_list:
        logger.error("没有可用的股票数据")
        return

    # 特征提取器
    feature_extractor = EnhancedFeatureExtractor(prediction_period=args.prediction_period)

    # 构建数据集
    train_dataset, val_dataset, test_dataset, dataset_builder = build_datasets(
        stock_data_list,
        feature_extractor,
        seq_len=args.seq_len,
        prediction_period=args.prediction_period,
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
    )

    # 获取特征维度
    sample_x, _ = train_dataset[0]
    input_size = sample_x.shape[1]
    logger.info(f"特征维度: {input_size}")

    # 创建模型
    model = create_lstm_model(
        input_size=input_size,
        device=device,
    )

    # 创建训练器
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
    )

    # 训练
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.output_dir,
    )

    # 评估测试集
    if test_loader:
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"测试集指标: {test_metrics}")
    else:
        test_metrics = {}

    # 保存模型和相关信息
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存scaler
    dataset_builder.save_scaler(str(output_dir / "lstm_scaler.pkl"))

    # 保存特征列
    feature_cols = dataset_builder.feature_cols
    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"保存特征列: {len(feature_cols)} 个")

    # 保存模型信息
    model_info = {
        "model_name": "LSTM Classifier",
        "model_version": "1.0.0",
        "model_path": str(output_dir / "lstm_best.pt"),
        "feature_count": input_size,
        "seq_len": args.seq_len,
        "prediction_period": args.prediction_period,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "test_samples": len(test_dataset) if test_dataset else 0,
        "train_auc": history["val_auc"][-1] if history["val_auc"] else 0,
        "test_auc": test_metrics.get("auc", 0),
        "test_accuracy": test_metrics.get("accuracy", 0),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info("训练完成!")
    logger.info(f"最佳验证AUC: {trainer.best_val_auc:.4f}")
    logger.info(f"模型保存至: {output_dir}")


if __name__ == "__main__":
    main()
