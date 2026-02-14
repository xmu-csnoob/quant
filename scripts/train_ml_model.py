"""
ML模型训练流水线

功能：
- 数据准备和特征提取
- 模型训练（XGBoost）
- 超参数调优（可选）
- 模型评估和验证
- 模型版本管理
"""

import sys
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from loguru import logger

from src.data.storage.sqlite_storage import SQLiteStorage
from src.utils.features.enhanced_features import EnhancedFeatureExtractor


class MLTrainingPipeline:
    """ML模型训练流水线"""

    def __init__(
        self,
        prediction_period: int = 5,
        min_samples: int = 100,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        初始化训练流水线

        Args:
            prediction_period: 预测周期（天数）
            min_samples: 最小样本数
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.prediction_period = prediction_period
        self.min_samples = min_samples
        self.test_size = test_size
        self.random_state = random_state

        self.storage = SQLiteStorage()
        self.feature_extractor = EnhancedFeatureExtractor(prediction_period=prediction_period)

        # 模型参数
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'error'],
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': random_state,
            'n_jobs': -1,
        }

        self.model = None
        self.feature_cols: Optional[list] = None
        self.model_info: Dict[str, Any] = {}

    def prepare_data(
        self,
        start_date: str = "20180101",
        end_date: str = "20241231",
        max_stocks: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        准备训练数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            max_stocks: 最大股票数（用于测试）

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("=" * 60)
        logger.info("准备训练数据")
        logger.info("=" * 60)

        # 获取所有股票
        all_stocks = self.storage.get_all_stocks()
        if max_stocks:
            all_stocks = all_stocks[:max_stocks]

        logger.info(f"股票数量: {len(all_stocks)}")

        # 收集特征数据
        all_features = []
        success_count = 0

        for i, ts_code in enumerate(all_stocks):
            if (i + 1) % 50 == 0:
                logger.info(f"处理进度: {i+1}/{len(all_stocks)}")

            try:
                # 获取数据
                df = self.storage.get_daily_prices(ts_code, start_date, end_date)

                if df is None or len(df) < self.min_samples:
                    continue

                # 提取特征
                features = self.feature_extractor.extract(df)

                if features.empty:
                    continue

                # 计算标签：未来N天收益率是否为正
                features['future_return'] = features['close'].pct_change(self.prediction_period).shift(-self.prediction_period)
                features['label'] = (features['future_return'] > 0).astype(int)
                features['ts_code'] = ts_code

                all_features.append(features)
                success_count += 1

            except Exception as e:
                logger.warning(f"处理 {ts_code} 失败: {e}")
                continue

        logger.info(f"成功处理股票: {success_count}/{len(all_stocks)}")

        if not all_features:
            raise ValueError("没有有效的训练数据")

        # 合并所有数据
        data = pd.concat(all_features, ignore_index=True)

        # 获取特征列
        self.feature_cols = [c for c in data.columns if c.startswith('f_')]
        logger.info(f"特征数量: {len(self.feature_cols)}")

        # 删除包含NaN的行
        data = data.dropna(subset=self.feature_cols + ['label'])
        logger.info(f"有效样本数: {len(data)}")

        # 按时间分割（避免前视偏差）
        data = data.sort_values('trade_date')
        split_idx = int(len(data) * (1 - self.test_size))

        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        X_train = train_data[self.feature_cols]
        y_train = train_data['label']
        X_test = test_data[self.feature_cols]
        y_test = test_data['label']

        logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
        logger.info(f"正样本比例 - 训练: {y_train.mean():.2%}, 测试: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签
            params: 模型参数

        Returns:
            评估指标
        """
        logger.info("=" * 60)
        logger.info("训练模型")
        logger.info("=" * 60)

        # 使用提供的参数或默认参数
        train_params = params or self.default_params

        logger.info(f"模型参数: {json.dumps(train_params, indent=2)}")

        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, 'train')]
        if X_test is not None and y_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test)
            evals.append((dtest, 'eval'))

        # 训练模型
        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=train_params.get('n_estimators', 200),
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=20,
        )

        logger.info(f"最佳迭代: {self.model.best_iteration}")

        # 评估
        metrics = {}
        if X_test is not None and y_test is not None:
            metrics = self.evaluate(X_test, y_test)

        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估指标
        """
        logger.info("=" * 60)
        logger.info("评估模型")
        logger.info("=" * 60)

        dtest = xgb.DMatrix(X_test)
        y_prob = self.model.predict(dtest)
        y_pred = (y_prob > 0.5).astype(int)

        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob),
        }

        logger.info("评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # 分类报告
        logger.info("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['下跌', '上涨']))

        # 混淆矩阵
        logger.info("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  真负例(TN): {cm[0, 0]}")
        print(f"  假正例(FP): {cm[0, 1]}")
        print(f"  假负例(FN): {cm[1, 0]}")
        print(f"  真正例(TP): {cm[1, 1]}")

        # 预测概率分布
        logger.info("\n预测概率分布:")
        print(f"  最小值: {y_prob.min():.4f}")
        print(f"  最大值: {y_prob.max():.4f}")
        print(f"  平均值: {y_prob.mean():.4f}")
        print(f"  标准差: {y_prob.std():.4f}")

        return metrics

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 20,
    ) -> Dict:
        """
        超参数调优（简单网格搜索）

        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_trials: 尝试次数

        Returns:
            最佳参数
        """
        logger.info("=" * 60)
        logger.info(f"超参数调优 (尝试次数: {n_trials})")
        logger.info("=" * 60)

        # 参数搜索空间
        param_grid = {
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'learning_rate': [0.01, 0.05, 0.1],
        }

        # 使用时间序列分割
        tscv = TimeSeriesSplit(n_splits=3)

        best_score = 0
        best_params = self.default_params.copy()

        # 简单随机搜索
        import random
        random.seed(self.random_state)

        for trial in range(n_trials):
            # 随机选择参数
            trial_params = self.default_params.copy()
            for param, values in param_grid.items():
                trial_params[param] = random.choice(values)

            # 交叉验证
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                dtrain = xgb.DMatrix(X_tr, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_val)

                model = xgb.train(
                    trial_params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dval, 'val')],
                    verbose_eval=False,
                )

                y_prob = model.predict(dval)
                score = roc_auc_score(y_val, y_prob)
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)
            logger.info(f"Trial {trial+1}/{n_trials}: AUC = {mean_score:.4f}, params = {trial_params}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = trial_params

        logger.info(f"\n最佳AUC: {best_score:.4f}")
        logger.info(f"最佳参数: {json.dumps(best_params, indent=2)}")

        return best_params

    def save_model(self, model_dir: str = "models", version: Optional[str] = None):
        """
        保存模型

        Args:
            model_dir: 模型目录
            version: 版本号（可选）
        """
        if self.model is None:
            raise ValueError("模型未训练")

        logger.info("=" * 60)
        logger.info("保存模型")
        logger.info("=" * 60)

        # 创建目录
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # 生成版本号
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存模型文件
        model_file = model_path / f"xgboost_model_{version}.json"
        self.model.save_model(str(model_file))
        logger.info(f"模型文件: {model_file}")

        # 同时保存为默认文件名（供生产使用）
        default_model_file = model_path / "xgboost_model.json"
        self.model.save_model(str(default_model_file))
        logger.info(f"默认模型: {default_model_file}")

        # 保存特征列表
        feature_file = model_path / f"feature_cols_{version}.json"
        with open(feature_file, 'w') as f:
            json.dump(self.feature_cols, f)

        # 同时保存为默认文件
        default_feature_file = model_path / "feature_cols.json"
        with open(default_feature_file, 'w') as f:
            json.dump(self.feature_cols, f)
        logger.info(f"特征文件: {default_feature_file}")

        # 保存模型信息
        self.model_info = {
            'model_name': 'XGBoost Classifier',
            'model_version': version,
            'model_path': str(default_model_file),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'prediction_period': self.prediction_period,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'params': self.default_params,
        }

        info_file = model_path / f"model_info_{version}.json"
        with open(info_file, 'w') as f:
            json.dump(self.model_info, f, indent=2)

        # 同时保存为默认文件
        default_info_file = model_path / "model_info.json"
        with open(default_info_file, 'w') as f:
            json.dump(self.model_info, f, indent=2)
        logger.info(f"模型信息: {default_info_file}")

        logger.info("✓ 模型保存完成")

    def load_model(self, model_path: str = "models/xgboost_model.json"):
        """加载模型"""
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        logger.info(f"模型加载成功: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ML模型训练流水线")
    parser.add_argument('--start-date', type=str, default='20180101', help='开始日期')
    parser.add_argument('--end-date', type=str, default='20241231', help='结束日期')
    parser.add_argument('--max-stocks', type=int, default=None, help='最大股票数（用于测试）')
    parser.add_argument('--tune', action='store_true', help='是否进行超参数调优')
    parser.add_argument('--n-trials', type=int, default=20, help='超参数调优尝试次数')
    parser.add_argument('--version', type=str, default=None, help='模型版本号')
    parser.add_argument('--prediction-period', type=int, default=5, help='预测周期（天数）')

    args = parser.parse_args()

    # 初始化流水线
    pipeline = MLTrainingPipeline(
        prediction_period=args.prediction_period,
        min_samples=100,
        test_size=0.2,
    )

    # 准备数据
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        start_date=args.start_date,
        end_date=args.end_date,
        max_stocks=args.max_stocks,
    )

    # 超参数调优（可选）
    if args.tune:
        best_params = pipeline.hyperparameter_tuning(
            X_train, y_train,
            n_trials=args.n_trials,
        )
    else:
        best_params = None

    # 训练模型
    metrics = pipeline.train(
        X_train, y_train,
        X_test, y_test,
        params=best_params,
    )

    # 保存模型
    pipeline.save_model(version=args.version)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
