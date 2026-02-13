// 策略中心页面
import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Table, Tag, Button, Space, Switch, message, Statistic, Empty, Spin } from 'antd';
import {
  ThunderboltOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { strategyApi } from '../../api';
import type { Strategy, Signal } from '../../api/types';

const StrategyCenter: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
    // 减少轮询频率到30秒
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [strategiesData, signalsData] = await Promise.all([
        strategyApi.getList(),
        strategyApi.getSignals(),
      ]);
      setStrategies(strategiesData);
      setSignals(signalsData);
    } catch (error) {
      console.error('获取策略数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleStrategy = async (strategyId: string, currentStatus: string) => {
    try {
      const action = currentStatus === 'running' ? 'stop' : 'start';
      await strategyApi.toggleStrategy(strategyId, action);
      message.success(`策略已${action === 'start' ? '启动' : '停止'}`);
      fetchData();
    } catch (error) {
      message.error('操作失败');
    }
  };

  // 策略状态标签
  const getStatusTag = (status: string) => {
    const statusMap: Record<string, { color: string; icon: React.ReactNode; text: string }> = {
      running: { color: 'success', icon: <CheckCircleOutlined />, text: '运行中' },
      stopped: { color: 'default', icon: <CloseCircleOutlined />, text: '已停止' },
      error: { color: 'error', icon: <ExclamationCircleOutlined />, text: '异常' },
    };
    const config = statusMap[status] || statusMap.stopped;
    return (
      <Tag color={config.color} icon={config.icon}>
        {config.text}
      </Tag>
    );
  };

  // 信号表格列
  const signalColumns = [
    {
      title: '时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (time: string) => dayjs(time).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: '策略',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      width: 200,
      ellipsis: true,
    },
    {
      title: '股票代码',
      dataIndex: 'code',
      key: 'code',
      width: 120,
    },
    {
      title: '股票名称',
      dataIndex: 'name',
      key: 'name',
      width: 100,
    },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: 80,
      render: (direction: string) => (
        <Tag color={direction === 'buy' ? 'red' : 'green'}>
          {direction === 'buy' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      align: 'right' as const,
      width: 100,
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => (
        <Tag color={confidence >= 70 ? 'green' : confidence >= 50 ? 'orange' : 'red'}>
          {confidence.toFixed(0)}%
        </Tag>
      ),
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
      {/* 策略卡片 */}
      <Row gutter={[16, 16]}>
        {strategies.map((strategy) => (
          <Col xs={24} sm={12} lg={8} xl={6} key={strategy.id}>
            <Card
              hoverable
              title={
                <Space>
                  <ThunderboltOutlined style={{ color: '#1890ff' }} />
                  <span>{strategy.name}</span>
                </Space>
              }
              extra={getStatusTag(strategy.status)}
            >
              <p style={{ color: '#8c8c8c', marginBottom: 16, minHeight: 40 }}>
                {strategy.description}
              </p>

              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="收益率"
                    value={strategy.return_rate}
                    precision={2}
                    suffix="%"
                    valueStyle={{
                      fontSize: 18,
                      color: strategy.return_rate >= 0 ? '#cf1322' : '#3f8600',
                    }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="胜率"
                    value={strategy.win_rate}
                    precision={1}
                    suffix="%"
                    valueStyle={{ fontSize: 18 }}
                  />
                </Col>
              </Row>

              <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: '#8c8c8c' }}>
                  交易次数: {strategy.trade_count}
                </span>
                <Switch
                  checked={strategy.status === 'running'}
                  onChange={() => handleToggleStrategy(strategy.id, strategy.status)}
                  checkedChildren="运行"
                  unCheckedChildren="停止"
                />
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 实时信号 */}
      <Card
        title="实时信号"
        extra={
          <Button icon={<ReloadOutlined />} onClick={fetchData}>
            刷新
          </Button>
        }
        style={{ marginTop: 24 }}
      >
        {signals.length > 0 ? (
          <Table
            columns={signalColumns}
            dataSource={signals}
            rowKey="id"
            pagination={{ pageSize: 10 }}
            size="middle"
          />
        ) : (
          <Empty description="暂无信号" />
        )}
      </Card>
    </div>
  );
};

export default StrategyCenter;
