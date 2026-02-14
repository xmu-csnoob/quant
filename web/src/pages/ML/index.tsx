// ML预测页面
import React, { useEffect, useState } from 'react';
import {
  Row,
  Col,
  Card,
  Table,
  Tag,
  Button,
  Space,
  message,
  Statistic,
  Empty,
  Spin,
  Input,
  Tabs,
  Progress,
  Descriptions,
  Alert,
} from 'antd';
import {
  RobotOutlined,
  ReloadOutlined,
  SearchOutlined,
  RiseOutlined,
  FallOutlined,
  MinusOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { mlApi } from '../../api';
import type {
  MLStatus,
  MLModelInfo,
  MLPredictionStats,
  MLPredictionResponse,
  FeatureImportance,
} from '../../api/types';

const MLPredictionPage: React.FC = () => {
  const [status, setStatus] = useState<MLStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<MLModelInfo | null>(null);
  const [stats, setStats] = useState<MLPredictionStats | null>(null);
  const [topBuySignals, setTopBuySignals] = useState<MLPredictionResponse[]>([]);
  const [topSellSignals, setTopSellSignals] = useState<MLPredictionResponse[]>([]);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [searchCode, setSearchCode] = useState('');
  const [predictionResult, setPredictionResult] = useState<MLPredictionResponse | null>(null);

  useEffect(() => {
    fetchData();
    // 减少轮询频率到60秒，避免频繁请求
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [statusData, modelData, statsData, buySignals, sellSignals, features] = await Promise.all([
        mlApi.getStatus(),
        mlApi.getModelInfo().catch(() => null),
        mlApi.getStats().catch(() => null),
        mlApi.getTopSignals(20, 'buy').catch(() => []),
        mlApi.getTopSignals(20, 'sell').catch(() => []),
        mlApi.getFeatureImportance(20).catch(() => []),
      ]);
      setStatus(statusData);
      setModelInfo(modelData);
      setStats(statsData);
      setTopBuySignals(buySignals);
      setTopSellSignals(sellSignals);
      setFeatureImportance(features);
    } catch (error) {
      console.error('获取ML数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!searchCode.trim()) {
      message.warning('请输入股票代码');
      return;
    }

    setPredicting(true);
    try {
      const result = await mlApi.predictByCode(searchCode.trim());
      setPredictionResult(result);
      message.success('预测完成');
    } catch {
      message.error('预测失败，请检查股票代码是否正确');
    } finally {
      setPredicting(false);
    }
  };

  // 预测结果卡片
  const renderPredictionResult = () => {
    if (!predictionResult) return null;

    const { prediction, signal, probability, confidence } = predictionResult;
    const signalConfig: Record<string, { color: string; icon: React.ReactNode; text: string }> = {
      buy: { color: '#cf1322', icon: <RiseOutlined />, text: '买入' },
      sell: { color: '#3f8600', icon: <FallOutlined />, text: '卖出' },
      hold: { color: '#8c8c8c', icon: <MinusOutlined />, text: '观望' },
    };
    const config = signalConfig[signal] || signalConfig.hold;

    return (
      <Card
        title={
          <Space>
            <RobotOutlined style={{ color: '#1890ff' }} />
            <span>预测结果: {predictionResult.stock_name || predictionResult.ts_code}</span>
          </Space>
        }
        style={{ marginTop: 16 }}
      >
        <Row gutter={[24, 16]}>
          <Col span={24}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Tag
                color={config.color}
                style={{
                  fontSize: 18,
                  padding: '8px 24px',
                  borderRadius: 4,
                }}
              >
                {config.icon} {config.text}
              </Tag>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="上涨概率"
              value={probability * 100}
              precision={1}
              suffix="%"
              valueStyle={{
                color: probability >= 0.55 ? '#cf1322' : probability <= 0.45 ? '#3f8600' : '#8c8c8c',
              }}
            />
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="置信度"
              value={confidence * 100}
              precision={1}
              suffix="%"
            />
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="预测方向"
              value={prediction === 'up' ? '上涨' : prediction === 'down' ? '下跌' : '中性'}
            />
          </Col>
          <Col xs={12} sm={6}>
            <Statistic
              title="预测日期"
              value={predictionResult.trade_date}
            />
          </Col>
        </Row>
        <Progress
          percent={probability * 100}
          strokeColor={probability >= 0.5 ? '#cf1322' : '#3f8600'}
          format={() => `上涨概率: ${(probability * 100).toFixed(1)}%`}
          style={{ marginTop: 16 }}
        />
      </Card>
    );
  };

  // 特征重要性图表
  const getFeatureImportanceOption = () => {
    if (featureImportance.length === 0) return {};

    const sortedFeatures = [...featureImportance].reverse();
    return {
      title: {
        text: '特征重要性 Top 20',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: '重要性',
      },
      yAxis: {
        type: 'category',
        data: sortedFeatures.map(f => f.feature_name.replace('f_', '')),
        axisLabel: {
          width: 100,
          overflow: 'truncate',
        },
      },
      series: [
        {
          name: '重要性',
          type: 'bar',
          data: sortedFeatures.map(f => f.importance_score),
          itemStyle: {
            color: '#1890ff',
          },
        },
      ],
    };
  };

  // 信号表格列
  const signalColumns = [
    {
      title: '股票代码',
      dataIndex: 'ts_code',
      key: 'ts_code',
      width: 120,
    },
    {
      title: '股票名称',
      dataIndex: 'stock_name',
      key: 'stock_name',
      width: 100,
      render: (name: string | null, record: MLPredictionResponse) => name || record.ts_code,
    },
    {
      title: '信号',
      dataIndex: 'signal',
      key: 'signal',
      width: 80,
      render: (signal: string) => {
        const config: Record<string, { color: string; text: string }> = {
          buy: { color: 'red', text: '买入' },
          sell: { color: 'green', text: '卖出' },
          hold: { color: 'default', text: '观望' },
        };
        const { color, text } = config[signal] || config.hold;
        return <Tag color={color}>{text}</Tag>;
      },
    },
    {
      title: '上涨概率',
      dataIndex: 'probability',
      key: 'probability',
      width: 100,
      align: 'right' as const,
      render: (prob: number) => (
        <span style={{ color: prob >= 0.5 ? '#cf1322' : '#3f8600' }}>
          {(prob * 100).toFixed(1)}%
        </span>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      align: 'right' as const,
      render: (conf: number) => (
        <Progress
          percent={conf * 100}
          size="small"
          showInfo={false}
          strokeColor={conf >= 0.3 ? '#52c41a' : '#faad14'}
        />
      ),
    },
    {
      title: '日期',
      dataIndex: 'trade_date',
      key: 'trade_date',
      width: 120,
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div>
      {/* 模型状态 */}
      {!status?.model_loaded && (
        <Alert
          message="ML模型未加载"
          description="请先训练模型或检查模型文件是否存在"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        {/* 模型信息 */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <RobotOutlined style={{ color: '#1890ff' }} />
                <span>模型信息</span>
              </Space>
            }
            extra={
              <Button icon={<ReloadOutlined />} onClick={fetchData}>
                刷新
              </Button>
            }
          >
            {modelInfo ? (
              <Descriptions column={2} size="small">
                <Descriptions.Item label="模型名称">{modelInfo.model_name}</Descriptions.Item>
                <Descriptions.Item label="版本">{modelInfo.model_version}</Descriptions.Item>
                <Descriptions.Item label="特征数">{modelInfo.feature_count}</Descriptions.Item>
                <Descriptions.Item label="预测周期">{modelInfo.prediction_period}天</Descriptions.Item>
                <Descriptions.Item label="训练样本">{modelInfo.train_samples.toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="测试样本">{modelInfo.test_samples.toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="训练AUC">{(modelInfo.train_auc * 100).toFixed(1)}%</Descriptions.Item>
                <Descriptions.Item label="测试AUC">{(modelInfo.test_auc * 100).toFixed(1)}%</Descriptions.Item>
              </Descriptions>
            ) : (
              <Empty description="模型信息不可用" />
            )}
          </Card>
        </Col>

        {/* 预测统计 */}
        <Col xs={24} lg={12}>
          <Card
            title={
              <Space>
                <BarChartOutlined style={{ color: '#52c41a' }} />
                <span>预测统计</span>
              </Space>
            }
          >
            {stats ? (
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="胜率"
                    value={stats.win_rate * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: stats.win_rate >= 0.5 ? '#3f8600' : '#cf1322' }}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="盈亏比"
                    value={stats.profit_loss_ratio}
                    precision={2}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="平均收益"
                    value={stats.avg_return * 100}
                    precision={2}
                    suffix="%"
                  />
                </Col>
                <Col span={8} style={{ marginTop: 16 }}>
                  <Statistic
                    title="买入信号"
                    value={stats.buy_signals}
                    valueStyle={{ color: '#cf1322' }}
                  />
                </Col>
                <Col span={8} style={{ marginTop: 16 }}>
                  <Statistic
                    title="卖出信号"
                    value={stats.sell_signals}
                    valueStyle={{ color: '#3f8600' }}
                  />
                </Col>
                <Col span={8} style={{ marginTop: 16 }}>
                  <Statistic
                    title="观望信号"
                    value={stats.hold_signals}
                  />
                </Col>
              </Row>
            ) : (
              <Empty description="统计数据不可用" />
            )}
          </Card>
        </Col>
      </Row>

      {/* 股票预测 */}
      <Card
        title={
          <Space>
            <SearchOutlined style={{ color: '#1890ff' }} />
            <span>股票预测</span>
          </Space>
        }
        style={{ marginTop: 16 }}
      >
        <Space.Compact style={{ width: '100%', maxWidth: 500 }}>
          <Input
            placeholder="输入股票代码，如 600519.SH"
            value={searchCode}
            onChange={(e) => setSearchCode(e.target.value)}
            onPressEnter={handlePredict}
          />
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            loading={predicting}
            onClick={handlePredict}
          >
            预测
          </Button>
        </Space.Compact>
        {renderPredictionResult()}
      </Card>

      {/* TOP信号 */}
      <Card style={{ marginTop: 16 }}>
        <Tabs
          defaultActiveKey="buy"
          items={[
            {
              key: 'buy',
              label: (
                <span>
                  <RiseOutlined style={{ color: '#cf1322' }} />
                  TOP买入信号
                </span>
              ),
              children: (
                <Table
                  columns={signalColumns}
                  dataSource={topBuySignals}
                  rowKey="ts_code"
                  pagination={{ pageSize: 10 }}
                  size="middle"
                />
              ),
            },
            {
              key: 'sell',
              label: (
                <span>
                  <FallOutlined style={{ color: '#3f8600' }} />
                  TOP卖出信号
                </span>
              ),
              children: (
                <Table
                  columns={signalColumns}
                  dataSource={topSellSignals}
                  rowKey="ts_code"
                  pagination={{ pageSize: 10 }}
                  size="middle"
                />
              ),
            },
          ]}
        />
      </Card>

      {/* 特征重要性 */}
      {featureImportance.length > 0 && (
        <Card style={{ marginTop: 16 }}>
          <ReactECharts option={getFeatureImportanceOption()} style={{ height: 400 }} />
        </Card>
      )}
    </div>
  );
};

export default MLPredictionPage;
