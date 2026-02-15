"""
LSTM股票涨跌预测模型

3层LSTM网络架构：
- 输入: (batch, seq_len=20, features=60)
- LSTM层: 60 -> 128 -> 64 -> 32
- 全连接层: 32 -> 16 -> 8 -> 1
- 输出: 涨跌概率 [0, 1]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from loguru import logger


class StockLSTMClassifier(nn.Module):
    """
    LSTM股票涨跌分类器

    架构:
        输入层 (60个特征, 20天序列)
            ↓
        [BatchNorm1d]
            ↓
        [LSTM Layer 1] 60 → 128, dropout=0.2
            ↓
        [LSTM Layer 2] 128 → 64, dropout=0.2
            ↓
        [LSTM Layer 3] 64 → 32
            ↓
        [取最后时间步输出]
            ↓
        [FC Layer 1] 32 → 16, ReLU, BatchNorm, Dropout(0.3)
            ↓
        [FC Layer 2] 16 → 8, ReLU, BatchNorm, Dropout(0.3)
            ↓
        [Output] 8 → 1, Sigmoid
            ↓
        输出: 涨跌概率 [0, 1]
    """

    def __init__(
        self,
        input_size: int = 60,
        hidden_sizes: Tuple[int, int, int] = (128, 64, 32),
        fc_sizes: Tuple[int, int] = (16, 8),
        num_layers: int = 3,
        dropout: float = 0.2,
        fc_dropout: float = 0.3,
    ):
        """
        初始化LSTM模型

        Args:
            input_size: 输入特征维度
            hidden_sizes: LSTM各层隐藏层大小 (h1, h2, h3)
            fc_sizes: 全连接层大小 (fc1, fc2)
            num_layers: LSTM层数
            dropout: LSTM层间dropout
            fc_dropout: 全连接层dropout
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers

        # 输入批归一化
        self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM层
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0],
            hidden_size=hidden_sizes[1],
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(
            input_size=hidden_sizes[1],
            hidden_size=hidden_sizes[2],
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # 全连接层
        self.fc1 = nn.Linear(hidden_sizes[2], fc_sizes[0])
        self.bn1 = nn.BatchNorm1d(fc_sizes[0])
        self.relu1 = nn.ReLU()
        self.fc_dropout1 = nn.Dropout(fc_dropout)

        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.bn2 = nn.BatchNorm1d(fc_sizes[1])
        self.relu2 = nn.ReLU()
        self.fc_dropout2 = nn.Dropout(fc_dropout)

        # 输出层
        self.output = nn.Linear(fc_sizes[1], 1)
        self.sigmoid = nn.Sigmoid()

        # 初始化权重
        self._init_weights()

        logger.info(
            f"StockLSTMClassifier initialized: input={input_size}, "
            f"hidden={hidden_sizes}, fc={fc_sizes}, dropout={dropout}"
        )

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # LSTM forget gate bias = 1
                if 'lstm' in name:
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入张量 (batch, seq_len, features)
            hidden: 初始隐藏状态 (可选)

        Returns:
            output: 预测概率 (batch, 1)
            hidden: 最终隐藏状态
        """
        batch_size, seq_len, _ = x.size()

        # 输入归一化 (batch, seq_len, features) -> (batch, features, seq_len) -> BN -> 还原
        x_flat = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x_flat = self.input_bn(x_flat)
        x = x_flat.permute(0, 2, 1)  # (batch, seq_len, features)

        # LSTM层
        h1, c1 = hidden if hidden else (None, None)
        out, (h1, c1) = self.lstm1(x, (h1, c1) if h1 is not None else None)
        out = self.dropout1(out)

        out, (h2, c2) = self.lstm2(out)
        out = self.dropout2(out)

        out, (h3, c3) = self.lstm3(out)

        # 取最后时间步的输出
        out = out[:, -1, :]  # (batch, hidden_sizes[2])

        # 全连接层
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc_dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc_dropout2(out)

        # 输出层
        out = self.output(out)
        out = self.sigmoid(out)

        return out, (h3, c3)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测（不返回隐藏状态）

        Args:
            x: 输入张量 (batch, seq_len, features)

        Returns:
            预测概率 (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
        return output

    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_lstm_model(
    input_size: int = 60,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> StockLSTMClassifier:
    """
    创建LSTM模型

    Args:
        input_size: 输入特征维度
        device: 设备
        **kwargs: 其他模型参数

    Returns:
        LSTM模型
    """
    model = StockLSTMClassifier(input_size=input_size, **kwargs)
    model = model.to(device)
    logger.info(f"模型创建完成，参数数量: {model.count_parameters():,}")
    return model


def load_lstm_model(
    model_path: str,
    input_size: int = 60,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> StockLSTMClassifier:
    """
    加载训练好的LSTM模型

    Args:
        model_path: 模型文件路径
        input_size: 输入特征维度
        device: 设备
        **kwargs: 其他模型参数

    Returns:
        加载的模型
    """
    model = create_lstm_model(input_size=input_size, device=device, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"模型加载完成: {model_path}")
    return model
