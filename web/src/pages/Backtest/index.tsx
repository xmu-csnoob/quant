// 回测分析页面
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Form, DatePicker, Select, InputNumber, Button, message, Statistic, Table, Spin, Alert, Descriptions, Divider } from 'antd';
import { PlayCircleOutlined, RobotOutlined, InfoCircleOutlined, WarningOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs, { Dayjs } from 'dayjs';
import { backtestApi, mlApi, dataApi } from '../../api';
import type { BacktestConfig, BacktestResult, MLModelInfo, FeatureImportance } from '../../api/types';

const { RangePicker } = DatePicker;

const Backtest: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState('ma_macd_rsi');
  const [mlModelInfo, setMlModelInfo] = useState<MLModelInfo | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [mlLoading, setMlLoading] = useState(false);
  // 新增：日期范围状态
  const [dateRange, setDateRange] = useState<{
    minDate: Dayjs | null;
    maxDate: Dayjs | null;
    available: boolean;
    loading: boolean;
  }>({
    minDate: null,
    maxDate: null,
    available: false,
    loading: true,
  });

  // 加载日期范围
  useEffect(() => {
    loadDateRange();
  }, []);

  const loadDateRange = async () => {
    try {
      const range = await dataApi.getDateRange();
      if (range.available && range.min_date && range.max_date) {
        setDateRange({
          minDate: dayjs(range.min_date),
          maxDate: dayjs(range.max_date),
          available: true,
          loading: false,
        });
        // 设置默认日期范围为数据库中的范围
        form.setFieldsValue({
          date_range: [dayjs(range.min_date), dayjs(range.max_date)],
        });
      } else {
        setDateRange({
          minDate: null,
          maxDate: null,
          available: false,
          loading: false,
        });
      }
    } catch (error) {
      console.error('加载日期范围失败:', error);
      setDateRange({
        minDate: null,
        maxDate: null,
        available: false,
        loading: false,
      });
    }
  };

  // 加载ML模型信息
  useEffect(() => {
    if (selectedStrategy === 'ml_strategy') {
      loadMLInfo();
    }
  }, [selectedStrategy]);

  const loadMLInfo = async () => {
    setMlLoading(true);
    try {
      const [info, features] = await Promise.all([
        mlApi.getModelInfo().catch(() => null),
        mlApi.getFeatureImportance(10).catch(() => []),
      ]);
      setMlModelInfo(info);
      setFeatureImportance(features);
    } catch (error) {
      console.error('加载ML信息失败:', error);
    } finally {
      setMlLoading(false);
    }
  };

  // 策略列表（聚焦ML策略）
  const strategies = [
    { value: 'ml_strategy', label: '🤖 机器学习策略 (LSTM)' },
  ];

  const runBacktest = async (values: any) => {
    setLoading(true);
    try {
      const config: BacktestConfig = {
        strategy_id: values.strategy,
        start_date: values.date_range[0].format('YYYY-MM-DD'),
        end_date: values.date_range[1].format('YYYY-MM-DD'),
        initial_capital: values.initial_capital,
        commission_rate: values.commission_rate / 1000,
        slippage_rate: values.slippage_rate / 1000,
      };

      const { backtest_id } = await backtestApi.run(config);

      // 轮询获取结果
      let retryCount = 0;
      const pollResult = async () => {
        try {
          const res = await backtestApi.getResult(backtest_id);
          if (res.status === 'completed') {
            setResult(res);
            message.success('回测完成');
            setLoading(false);
          } else if (retryCount < 10) {
            retryCount++;
            setTimeout(pollResult, 1000);
          } else {
            message.error('回测超时');
            setLoading(false);
          }
        } catch {
          if (retryCount < 10) {
            retryCount++;
            setTimeout(pollResult, 1000);
          } else {
            message.error('获取回测结果失败');
            setLoading(false);
          }
        }
      };

      pollResult();
    } catch {
      message.error('启动回测失败');
      setLoading(false);
    }
  };

  // 净值曲线配置
  const getEquityCurveOption = () => {
    if (!result?.equity_curve) return {};

    const dates = result.equity_curve.map((p) => p.date);
    const values = result.equity_curve.map((p) => p.equity);
    const returns = result.equity_curve.map((p) => p.return_rate);

    return {
      title: { text: '净值曲线', left: 'center' },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const data = params[0];
          return `${data.axisValue}<br/>净值: ¥${data.value.toLocaleString()}<br/>收益率: ${returns[params[0].dataIndex].toFixed(2)}%`;
        },
      },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: dates, boundaryGap: false },
      yAxis: [
        {
          type: 'value',
          name: '净值',
          axisLabel: { formatter: (v: number) => `¥${(v / 10000).toFixed(0)}万` },
        },
        {
          type: 'value',
          name: '收益率',
          axisLabel: { formatter: '{value}%' },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: '净值',
          type: 'line',
          data: values,
          smooth: true,
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(24, 144, 255, 0.3)' },
                { offset: 1, color: 'rgba(24, 144, 255, 0.05)' },
              ],
            },
          },
          lineStyle: { color: '#1890ff', width: 2 },
        },
      ],
    };
  };

  // 特征重要性图表
  const getFeatureImportanceOption = () => {
    if (featureImportance.length === 0) return {};

    const sortedFeatures = [...featureImportance].reverse();
    return {
      title: { text: '特征重要性', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: { type: 'value', name: '重要性' },
      yAxis: {
        type: 'category',
        data: sortedFeatures.map(f => f.feature_name.replace('f_', '')),
        axisLabel: { width: 80, overflow: 'truncate' },
      },
      series: [
        {
          name: '重要性',
          type: 'bar',
          data: sortedFeatures.map(f => f.importance_score),
          itemStyle: { color: '#52c41a' },
        },
      ],
    };
  };

  // 交易记录表格列
  const tradeColumns = [
    { title: '日期', dataIndex: 'date', key: 'date', width: 120 },
    { title: '代码', dataIndex: 'code', key: 'code', width: 120 },
    { title: '名称', dataIndex: 'name', key: 'name', width: 100 },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: 80,
      render: (d: string) => (
        <span style={{ color: d === 'buy' ? '#cf1322' : '#3f8600' }}>
          {d === 'buy' ? '买入' : '卖出'}
        </span>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      align: 'right' as const,
      render: (p: number) => `¥${p.toFixed(2)}`,
    },
    {
      title: '数量',
      dataIndex: 'shares',
      key: 'shares',
      align: 'right' as const,
      render: (s: number) => s.toLocaleString(),
    },
    {
      title: '盈亏',
      dataIndex: 'profit',
      key: 'profit',
      align: 'right' as const,
      render: (p: number) => (
        <span style={{ color: p >= 0 ? '#cf1322' : '#3f8600' }}>
          {p >= 0 ? '+' : ''}¥{p.toFixed(2)}
        </span>
      ),
    },
  ];

  // ML模型信息卡片
  const renderMLModelInfo = () => {
    if (selectedStrategy !== 'ml_strategy') return null;

    if (mlLoading) {
      return (
        <Card style={{ marginTop: 16 }}>
          <Spin />
        </Card>
      );
    }

    if (!mlModelInfo) {
      return (
        <Alert
          message="ML模型未加载"
          description="请先训练模型或检查模型文件是否存在"
          type="warning"
          showIcon
          icon={<InfoCircleOutlined />}
          style={{ marginTop: 16 }}
        />
      );
    }

    return (
      <Card
        title={
          <span>
            <RobotOutlined style={{ marginRight: 8, color: '#1890ff' }} />
            ML模型信息
          </span>
        }
        style={{ marginTop: 16 }}
        size="small"
      >
        <Descriptions column={2} size="small">
          <Descriptions.Item label="模型">{mlModelInfo.model_name}</Descriptions.Item>
          <Descriptions.Item label="版本">{mlModelInfo.model_version}</Descriptions.Item>
          <Descriptions.Item label="特征数">{mlModelInfo.feature_count}</Descriptions.Item>
          <Descriptions.Item label="预测周期">{mlModelInfo.prediction_period}天</Descriptions.Item>
          <Descriptions.Item label="训练AUC">{(mlModelInfo.train_auc * 100).toFixed(1)}%</Descriptions.Item>
          <Descriptions.Item label="测试AUC">{(mlModelInfo.test_auc * 100).toFixed(1)}%</Descriptions.Item>
        </Descriptions>

        {featureImportance.length > 0 && (
          <>
            <Divider style={{ margin: '12px 0' }} />
            <ReactECharts option={getFeatureImportanceOption()} style={{ height: 200 }} />
          </>
        )}
      </Card>
    );
  };

  return (
    <div>
      <Row gutter={24}>
        {/* 回测设置面板 */}
        <Col xs={24} lg={8}>
          <Card title="回测设置">
            <Form
              form={form}
              layout="vertical"
              onFinish={runBacktest}
              initialValues={{
                strategy: 'ma_macd_rsi',
                date_range: [dayjs().subtract(1, 'year'), dayjs()],
                initial_capital: 1000000,
                commission_rate: 0.3,
                slippage_rate: 1,
              }}
            >
              <Form.Item
                name="strategy"
                label="选择策略"
                rules={[{ required: true, message: '请选择策略' }]}
              >
                <Select
                  options={strategies}
                  onChange={(value) => setSelectedStrategy(value)}
                />
              </Form.Item>

              <Form.Item
                name="date_range"
                label="回测区间"
                rules={[{ required: true, message: '请选择回测区间' }]}
                extra={dateRange.available && dateRange.minDate && dateRange.maxDate ?
                  `可用数据范围: ${dateRange.minDate.format('YYYY-MM-DD')} ~ ${dateRange.maxDate.format('YYYY-MM-DD')}` :
                  '暂无数据，请先导入数据'
                }
              >
                <RangePicker
                  style={{ width: '100%' }}
                  disabled={!dateRange.available}
                  disabledDate={(current: Dayjs) => {
                    if (!dateRange.minDate || !dateRange.maxDate) return true;
                    // 禁用超出数据范围的日期
                    return current && (current < dateRange.minDate || current > dateRange.maxDate);
                  }}
                />
              </Form.Item>

              <Form.Item
                name="initial_capital"
                label="初始资金"
                rules={[{ required: true, message: '请输入初始资金' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  min={10000}
                  step={10000}
                  formatter={(v) => `¥ ${v}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  parser={(v) => v!.replace(/¥\s?|(,*)/g, '') as any}
                />
              </Form.Item>

              <Form.Item
                name="commission_rate"
                label="手续费率 (‰)"
                tooltip="买入0.03%，卖出0.13%（含印花税）"
              >
                <InputNumber style={{ width: '100%' }} min={0} max={10} step={0.1} precision={2} />
              </Form.Item>

              <Form.Item
                name="slippage_rate"
                label="滑点率 (‰)"
              >
                <InputNumber style={{ width: '100%' }} min={0} max={10} step={0.5} precision={1} />
              </Form.Item>

              <Form.Item>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<PlayCircleOutlined />}
                  loading={loading}
                  disabled={!dateRange.available}
                  block
                >
                  开始回测
                </Button>
              </Form.Item>
            </Form>

            {/* 无数据警告 */}
            {!dateRange.loading && !dateRange.available && (
              <Alert
                message="暂无可用数据"
                description={
                  <span>
                    请先导入股票数据。
                    <a href="https://github.com/xmu-csnoob/quant/blob/main/scripts/import_data.py" target="_blank" rel="noopener noreferrer">
                      查看数据导入脚本
                    </a>
                  </span>
                }
                type="warning"
                showIcon
                icon={<WarningOutlined />}
                style={{ marginTop: 16 }}
              />
            )}
          </Card>

          {/* ML模型信息 */}
          {renderMLModelInfo()}
        </Col>

        {/* 回测结果面板 */}
        <Col xs={24} lg={16}>
          {loading && (
            <Card>
              <div style={{ textAlign: 'center', padding: 100 }}>
                <Spin size="large" />
                <p style={{ marginTop: 16, color: '#8c8c8c' }}>回测运行中...</p>
              </div>
            </Card>
          )}

          {result && !loading && (
            <>
              {/* 统计指标 */}
              <Card style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="总收益率"
                      value={result.total_return}
                      precision={2}
                      suffix="%"
                      valueStyle={{ color: result.total_return >= 0 ? '#cf1322' : '#3f8600' }}
                    />
                  </Col>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="年化收益"
                      value={result.annual_return}
                      precision={2}
                      suffix="%"
                    />
                  </Col>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="最大回撤"
                      value={result.max_drawdown}
                      precision={2}
                      suffix="%"
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="夏普比率"
                      value={result.sharpe_ratio}
                      precision={2}
                    />
                  </Col>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="胜率"
                      value={result.win_rate}
                      precision={1}
                      suffix="%"
                    />
                  </Col>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="盈亏比"
                      value={result.profit_factor}
                      precision={2}
                    />
                  </Col>
                  <Col xs={12} sm={8} md={6}>
                    <Statistic
                      title="交易次数"
                      value={result.trade_count}
                    />
                  </Col>
                  {/* T+1统计 */}
                  {(result.t1_violations > 0 || result.t1_skipped_sells > 0) && (
                    <>
                      <Col xs={12} sm={8} md={6}>
                        <Statistic
                          title="T+1违规尝试"
                          value={result.t1_violations || 0}
                          valueStyle={{ color: '#faad14' }}
                        />
                      </Col>
                      <Col xs={12} sm={8} md={6}>
                        <Statistic
                          title="T+1跳过卖出"
                          value={result.t1_skipped_sells || 0}
                          valueStyle={{ color: '#faad14' }}
                        />
                      </Col>
                    </>
                  )}
                </Row>
              </Card>

              {/* 净值曲线 */}
              <Card style={{ marginBottom: 16 }}>
                <ReactECharts option={getEquityCurveOption()} style={{ height: 350 }} />
              </Card>

              {/* 交易记录 */}
              <Card title="交易记录">
                <Table
                  columns={tradeColumns}
                  dataSource={result.trades}
                  rowKey={(r) => `${r.date}-${r.code}-${r.direction}`}
                  pagination={{ pageSize: 10 }}
                  size="small"
                  scroll={{ x: 800 }}
                />
              </Card>
            </>
          )}

          {!result && !loading && (
            <Card>
              <div style={{ textAlign: 'center', padding: 100, color: '#8c8c8c' }}>
                <PlayCircleOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <p>请设置回测参数并开始回测</p>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default Backtest;
