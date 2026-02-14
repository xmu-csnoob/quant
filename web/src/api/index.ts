// API Client - API调用封装
import axios, { type AxiosInstance, type AxiosResponse } from 'axios';
import { message } from 'antd';
import type {
  AccountSummary,
  Position,
  Order,
  CreateOrderRequest,
  Strategy,
  Signal,
  BacktestConfig,
  BacktestResult,
  DataStatus,
  KlineData,
  ApiResponse,
  PaginatedResponse,
  RiskConfig,
  RiskStatus,
  MLPredictionRequest,
  MLPredictionResponse,
  MLModelInfo,
  MLPredictionStats,
  FeatureImportance,
  MLStatus,
} from './types';

// 创建axios实例
const client: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 响应拦截器
client.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    const errorMsg = error.response?.data?.message || error.message || '请求失败';
    message.error(errorMsg);
    return Promise.reject(error);
  }
);

// ==================== 账户API ====================
export const accountApi = {
  // 获取账户概览
  getSummary: async (): Promise<AccountSummary> => {
    const res = await client.get<ApiResponse<AccountSummary>>('/api/account/summary');
    return res.data.data;
  },

  // 获取持仓列表
  getPositions: async (): Promise<Position[]> => {
    const res = await client.get<ApiResponse<Position[]>>('/api/account/positions');
    return res.data.data;
  },
};

// ==================== 交易API ====================
export const tradingApi = {
  // 获取订单列表
  getOrders: async (params?: {
    status?: string;
    page?: number;
    page_size?: number;
  }): Promise<PaginatedResponse<Order>> => {
    const res = await client.get<ApiResponse<PaginatedResponse<Order>>>('/api/trading/orders', { params });
    return res.data.data;
  },

  // 创建订单
  createOrder: async (order: CreateOrderRequest): Promise<Order> => {
    const res = await client.post<ApiResponse<Order>>('/api/trading/orders', order);
    return res.data.data;
  },

  // 取消订单
  cancelOrder: async (orderId: string): Promise<void> => {
    await client.delete(`/api/trading/orders/${orderId}`);
  },
};

// ==================== 策略API ====================
export const strategyApi = {
  // 获取策略列表
  getList: async (): Promise<Strategy[]> => {
    const res = await client.get<ApiResponse<Strategy[]>>('/api/strategy/list');
    return res.data.data;
  },

  // 获取实时信号
  getSignals: async (): Promise<Signal[]> => {
    const res = await client.get<ApiResponse<Signal[]>>('/api/strategy/signals');
    return res.data.data;
  },

  // 启动/停止策略
  toggleStrategy: async (strategyId: string, action: 'start' | 'stop'): Promise<void> => {
    await client.post(`/api/strategy/${strategyId}/${action}`);
  },
};

// ==================== 回测API ====================
export const backtestApi = {
  // 启动回测
  run: async (config: BacktestConfig): Promise<{ backtest_id: string }> => {
    const res = await client.post<ApiResponse<{ backtest_id: string }>>('/api/backtest/run', config);
    return res.data.data;
  },

  // 获取回测结果
  getResult: async (backtestId: string): Promise<BacktestResult> => {
    const res = await client.get<ApiResponse<BacktestResult>>(`/api/backtest/results/${backtestId}`);
    return res.data.data;
  },
};

// ==================== 数据API ====================
export const dataApi = {
  // 获取数据状态
  getStatus: async (): Promise<DataStatus> => {
    const res = await client.get<ApiResponse<DataStatus>>('/api/data/status');
    return res.data.data;
  },

  // 获取可用日期范围
  getDateRange: async (): Promise<{
    min_date: string | null;
    max_date: string | null;
    total_records: number;
    available: boolean;
  }> => {
    const res = await client.get<ApiResponse<{
      min_date: string | null;
      max_date: string | null;
      total_records: number;
      available: boolean;
    }>>('/api/data/date-range');
    return res.data.data;
  },

  // 获取K线数据
  getKline: async (code: string, params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<KlineData[]> => {
    const res = await client.get<ApiResponse<KlineData[]>>(`/api/data/kline/${code}`, { params });
    return res.data.data;
  },
};

// ==================== 风控API ====================
export const riskApi = {
  // 获取风控配置
  getConfig: async (): Promise<RiskConfig> => {
    const res = await client.get<ApiResponse<RiskConfig>>('/api/risk/config');
    return res.data.data;
  },

  // 保存风控配置
  saveConfig: async (config: Partial<RiskConfig>): Promise<void> => {
    await client.post('/api/risk/config', config);
  },

  // 获取风控状态
  getStatus: async (): Promise<RiskStatus> => {
    const res = await client.get<ApiResponse<RiskStatus>>('/api/risk/status');
    return res.data.data;
  },
};

// WebSocket连接
export const createWebSocket = (path: string): WebSocket => {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  return new WebSocket(`${wsUrl}${path}`);
};

// ==================== ML预测API ====================
export const mlApi = {
  // 获取ML模型状态
  getStatus: async (): Promise<MLStatus> => {
    const res = await client.get<MLStatus>('/api/ml/status');
    return res.data;
  },

  // 获取模型信息
  getModelInfo: async (): Promise<MLModelInfo> => {
    const res = await client.get<ApiResponse<MLModelInfo>>('/api/ml/model/info');
    return res.data.data;
  },

  // 预测单只股票
  predict: async (request: MLPredictionRequest): Promise<MLPredictionResponse> => {
    const res = await client.post<ApiResponse<MLPredictionResponse>>('/api/ml/predict', request);
    return res.data.data;
  },

  // 根据股票代码预测（GET方式）
  predictByCode: async (tsCode: string): Promise<MLPredictionResponse> => {
    const res = await client.get<ApiResponse<MLPredictionResponse>>(`/api/ml/${tsCode}`);
    return res.data.data;
  },

  // 批量预测
  batchPredict: async (tsCodes: string[]): Promise<MLPredictionResponse[]> => {
    const res = await client.post<ApiResponse<MLPredictionResponse[]>>('/api/ml/predict/batch', { ts_codes: tsCodes });
    return res.data.data;
  },

  // 获取TOP信号
  getTopSignals: async (limit: number = 20, signalType: 'buy' | 'sell' = 'buy'): Promise<MLPredictionResponse[]> => {
    const res = await client.get<ApiResponse<MLPredictionResponse[]>>('/api/ml/signals/top', {
      params: { limit, signal_type: signalType },
    });
    return res.data.data;
  },

  // 获取特征重要性
  getFeatureImportance: async (topN: number = 20): Promise<FeatureImportance[]> => {
    const res = await client.get<ApiResponse<FeatureImportance[]>>('/api/ml/features/importance', {
      params: { top_n: topN },
    });
    return res.data.data;
  },

  // 获取预测统计
  getStats: async (): Promise<MLPredictionStats> => {
    const res = await client.get<ApiResponse<MLPredictionStats>>('/api/ml/stats');
    return res.data.data;
  },
};

export default client;
