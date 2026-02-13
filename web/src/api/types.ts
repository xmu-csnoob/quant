// API Types - API响应类型定义

// 账户相关
export interface AccountSummary {
  total_assets: number;      // 总资产
  cash: number;              // 现金
  market_value: number;      // 持仓市值
  total_profit: number;      // 总盈亏
  total_return: number;      // 总收益率
  today_profit: number;      // 今日盈亏
  today_return: number;      // 今日收益率
}

export interface Position {
  code: string;              // 股票代码
  name: string;              // 股票名称
  shares: number;            // 持仓数量
  available: number;         // 可用数量
  cost_price: number;        // 成本价
  current_price: number;     // 当前价
  market_value: number;      // 市值
  profit: number;            // 盈亏
  profit_ratio: number;      // 盈亏比例
  weight: number;            // 持仓权重
}

// 交易相关
export interface Order {
  order_id: string;          // 订单ID
  code: string;              // 股票代码
  name: string;              // 股票名称
  direction: 'buy' | 'sell'; // 买卖方向
  order_type: 'limit' | 'market'; // 订单类型
  price: number;             // 委托价格
  shares: number;            // 委托数量
  filled_shares: number;     // 成交数量
  status: string;            // 订单状态
  created_at: string;        // 创建时间
  updated_at: string;        // 更新时间
}

export interface CreateOrderRequest {
  code: string;
  direction: 'buy' | 'sell';
  order_type: 'limit' | 'market';
  price?: number;
  shares: number;
}

// 策略相关
export interface Strategy {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'stopped' | 'error';
  return_rate: number;
  win_rate: number;
  trade_count: number;
}

export interface Signal {
  id: string;
  strategy_id: string;
  strategy_name: string;
  code: string;
  name: string;
  direction: 'buy' | 'sell';
  price: number;
  confidence: number;
  created_at: string;
}

// 回测相关
export interface BacktestConfig {
  strategy_id: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_rate: number;
  slippage_rate: number;
}

export interface BacktestResult {
  id: string;
  status: 'running' | 'completed' | 'failed';
  config: BacktestConfig;

  // 统计指标
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
  profit_factor: number;
  trade_count: number;

  // T+1统计
  t1_violations: number;       // T+1违规尝试次数
  t1_skipped_sells: number;    // 因T+1跳过的卖出

  // 时间序列数据
  equity_curve: EquityPoint[];
  trades: Trade[];
}

export interface EquityPoint {
  date: string;
  equity: number;
  return_rate: number;
}

export interface Trade {
  date: string;
  code: string;
  name: string;
  direction: 'buy' | 'sell';
  price: number;
  shares: number;
  profit: number;
}

// 数据相关
export interface DataStatus {
  total_stocks: number;
  last_update: string;
  data_sources: string[];
  update_status: 'idle' | 'running' | 'error';
}

export interface KlineData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
}

// 风控相关
export interface RiskConfig {
  max_position_count: number;
  max_position_ratio: number;
  min_position_ratio: number;
  stop_loss_ratio: number;
  take_profit_ratio: number;
  enable_auto_stop_loss: boolean;
  enable_trailing_stop: boolean;
  trailing_stop_ratio: number;
  max_daily_loss: number;
  max_drawdown: number;
  enable_consecutive_loss: boolean;
  max_consecutive_losses: number;
  enable_t1_rule: boolean;
}

export interface RiskStatus {
  risk_level: 'low' | 'medium' | 'high';
  current_positions: number;
  position_ratio: number;
  daily_loss: number;
  daily_loss_ratio: number;
  max_drawdown: number;
  max_drawdown_ratio: number;
  consecutive_losses: number;
  t1_locked_shares: number;
}

// 通用响应
export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

// ML预测相关
export interface MLPredictionRequest {
  ts_code: string;
  include_features?: boolean;
}

export interface MLPredictionResponse {
  ts_code: string;
  stock_name: string | null;
  prediction: 'up' | 'down' | 'neutral';
  probability: number;
  confidence: number;
  predicted_return: number | null;
  signal: 'buy' | 'sell' | 'hold';
  features: Record<string, number> | null;
  trade_date: string;
  model_version: string;
  prediction_period: number;
}

export interface MLModelInfo {
  model_name: string;
  model_version: string;
  model_path: string;
  feature_count: number;
  prediction_period: number;
  train_samples: number;
  test_samples: number;
  train_auc: number;
  test_auc: number;
  train_accuracy: number;
  test_accuracy: number;
  created_at: string;
}

export interface MLPredictionStats {
  total_predictions: number;
  correct_predictions: number;
  accuracy: number;
  win_rate: number;
  avg_return: number;
  profit_loss_ratio: number;
  buy_signals: number;
  sell_signals: number;
  hold_signals: number;
}

export interface FeatureImportance {
  feature_name: string;
  importance_score: number;
  rank: number;
}

export interface MLStatus {
  model_loaded: boolean;
  model_path: string | null;
  feature_count: number;
}

export interface BatchPredictionRequest {
  ts_codes: string[];
}
