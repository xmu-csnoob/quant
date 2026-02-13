// 仪表盘页面
import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Statistic, Table, Progress, Spin, Empty } from 'antd';
import {
  RiseOutlined,
  FallOutlined,
  WalletOutlined,
  StockOutlined,
  TrophyOutlined,
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import { accountApi } from '../../api';
import { useAccountStore } from '../../stores';
import type { Position } from '../../api/types';

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const { summary, positions, setSummary, setPositions } = useAccountStore();

  useEffect(() => {
    fetchDashboardData();
    // 定时刷新数据
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [summaryData, positionsData] = await Promise.all([
        accountApi.getSummary(),
        accountApi.getPositions(),
      ]);
      setSummary(summaryData);
      setPositions(positionsData);
    } catch (error) {
      console.error('获取仪表盘数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 净值曲线图配置
  const getEquityCurveOption = () => {
    // 生成模拟净值曲线数据
    const dates = [];
    const values = [];
    let value = 1000000;

    for (let i = 30; i >= 0; i--) {
      dates.push(dayjs().subtract(i, 'day').format('MM-DD'));
      value = value * (1 + (Math.random() - 0.48) * 0.02);
      values.push(Math.round(value));
    }

    return {
      title: {
        text: '净值曲线',
        left: 'center',
        textStyle: { fontSize: 16 },
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const data = params[0];
          const change = data.value > 1000000;
          const percent = ((data.value - 1000000) / 1000000 * 100).toFixed(2);
          return `${data.axisValue}<br/>净值: ¥${data.value.toLocaleString()}<br/>收益率: ${change ? '+' : ''}${percent}%`;
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: dates,
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (value: number) => `¥${(value / 10000).toFixed(0)}万`,
        },
      },
      series: [
        {
          name: '净值',
          type: 'line',
          smooth: true,
          data: values,
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
          itemStyle: { color: '#1890ff' },
        },
      ],
    };
  };

  // 持仓分布饼图
  const getPositionPieOption = () => {
    if (!positions || positions.length === 0) {
      return {};
    }

    const data = positions.map((pos) => ({
      name: pos.name,
      value: pos.market_value,
    }));

    // 添加现金
    if (summary) {
      data.push({ name: '现金', value: summary.cash });
    }

    return {
      title: {
        text: '持仓分布',
        left: 'center',
        textStyle: { fontSize: 16 },
      },
      tooltip: {
        trigger: 'item',
        formatter: '{b}: ¥{c} ({d}%)',
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        top: 'middle',
      },
      series: [
        {
          type: 'pie',
          radius: ['40%', '70%'],
          center: ['60%', '50%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderRadius: 10,
            borderColor: '#fff',
            borderWidth: 2,
          },
          label: {
            show: false,
          },
          emphasis: {
            label: {
              show: true,
              fontSize: 14,
              fontWeight: 'bold',
            },
          },
          data,
        },
      ],
    };
  };

  // 持仓表格列定义
  const positionColumns = [
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
      title: '持仓数量',
      dataIndex: 'shares',
      key: 'shares',
      align: 'right' as const,
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: '可卖数量',
      dataIndex: 'available',
      key: 'available',
      align: 'right' as const,
      render: (value: number, record: Position) => {
        const locked = record.shares - value;
        return (
          <span>
            <span style={{ color: value > 0 ? '#262626' : '#cf1322' }}>
              {value.toLocaleString()}
            </span>
            {locked > 0 && (
              <span style={{ color: '#faad14', fontSize: 12, marginLeft: 4 }}>
                (锁定{locked})
              </span>
            )}
          </span>
        );
      },
    },
    {
      title: '成本价',
      dataIndex: 'cost_price',
      key: 'cost_price',
      align: 'right' as const,
      render: (value: number) => `¥${value.toFixed(2)}`,
    },
    {
      title: '当前价',
      dataIndex: 'current_price',
      key: 'current_price',
      align: 'right' as const,
      render: (value: number) => `¥${value.toFixed(2)}`,
    },
    {
      title: '市值',
      dataIndex: 'market_value',
      key: 'market_value',
      align: 'right' as const,
      render: (value: number) => `¥${value.toLocaleString()}`,
    },
    {
      title: '盈亏',
      dataIndex: 'profit',
      key: 'profit',
      align: 'right' as const,
      render: (value: number, record: Position) => (
        <span className={value >= 0 ? 'rise' : 'fall'}>
          {value >= 0 ? '+' : ''}¥{value.toLocaleString()}
          <span style={{ marginLeft: 8, fontSize: 12 }}>
            ({value >= 0 ? '+' : ''}{record.profit_ratio.toFixed(2)}%)
          </span>
        </span>
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
      {/* 统计卡片 */}
      <Row gutter={16}>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card">
            <Statistic
              title="总资产"
              value={summary?.total_assets || 0}
              precision={2}
              prefix={<WalletOutlined />}
              suffix="元"
              valueStyle={{ fontSize: 24 }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card">
            <Statistic
              title="今日收益"
              value={summary?.today_profit || 0}
              precision={2}
              prefix={summary?.today_profit && summary.today_profit >= 0 ? <RiseOutlined /> : <FallOutlined />}
              suffix="元"
              valueStyle={{
                color: summary?.today_profit && summary.today_profit >= 0 ? '#cf1322' : '#3f8600',
              }}
            />
            <Progress
              percent={Math.abs(summary?.today_return || 0)}
              size="small"
              showInfo={false}
              strokeColor={summary?.today_return && summary.today_return >= 0 ? '#cf1322' : '#3f8600'}
              style={{ marginTop: 8 }}
            />
            <span style={{ fontSize: 12, color: '#8c8c8c' }}>
              今日收益率: {summary?.today_return && summary.today_return >= 0 ? '+' : ''}{summary?.today_return?.toFixed(2) || 0}%
            </span>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card">
            <Statistic
              title="持仓市值"
              value={summary?.market_value || 0}
              precision={2}
              prefix={<StockOutlined />}
              suffix="元"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card className="stat-card">
            <Statistic
              title="累计收益"
              value={summary?.total_profit || 0}
              precision={2}
              prefix={<TrophyOutlined />}
              suffix="元"
              valueStyle={{
                color: summary?.total_profit && summary.total_profit >= 0 ? '#cf1322' : '#3f8600',
              }}
            />
            <div style={{ marginTop: 8, fontSize: 12, color: '#8c8c8c' }}>
              累计收益率: {summary?.total_return && summary.total_return >= 0 ? '+' : ''}{summary?.total_return?.toFixed(2) || 0}%
            </div>
          </Card>
        </Col>
      </Row>

      {/* 图表区 */}
      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col xs={24} lg={16}>
          <Card>
            <ReactECharts
              option={getEquityCurveOption()}
              style={{ height: 350 }}
              notMerge={true}
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card>
            {positions && positions.length > 0 ? (
              <ReactECharts
                option={getPositionPieOption()}
                style={{ height: 350 }}
                notMerge={true}
              />
            ) : (
              <Empty description="暂无持仓" style={{ padding: 50 }} />
            )}
          </Card>
        </Col>
      </Row>

      {/* 持仓列表 */}
      <Card title="当前持仓" style={{ marginTop: 16 }}>
        <Table
          columns={positionColumns}
          dataSource={positions}
          rowKey="code"
          pagination={false}
          size="middle"
        />
      </Card>
    </div>
  );
};

export default Dashboard;
