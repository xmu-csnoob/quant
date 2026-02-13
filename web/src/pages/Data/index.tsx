// 数据中心页面
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Tag, Button, Input, Space, Spin } from 'antd';
import { SearchOutlined, ReloadOutlined, DatabaseOutlined, CloudServerOutlined, CheckCircleOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { dataApi } from '../../api';
import type { KlineData } from '../../api/types';

const DataCenter: React.FC = () => {
  const [, setLoading] = useState(false);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [klineCode, setKlineCode] = useState('600519.SH');
  const [klineData, setKlineData] = useState<KlineData[]>([]);
  const [klineLoading, setKlineLoading] = useState(false);

  useEffect(() => {
    fetchDataStatus();
  }, []);

  useEffect(() => {
    if (klineCode) {
      fetchKlineData();
    }
  }, [klineCode]);

  const fetchDataStatus = async () => {
    setLoading(true);
    try {
      const status = await dataApi.getStatus();
      setDataStatus(status);
    } catch (error) {
      console.error('获取数据状态失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchKlineData = async () => {
    setKlineLoading(true);
    try {
      const data = await dataApi.getKline(klineCode);
      setKlineData(data);
    } catch (error) {
      console.error('获取K线数据失败:', error);
    } finally {
      setKlineLoading(false);
    }
  };

  // K线图配置
  const getKlineOption = () => {
    if (!klineData || klineData.length === 0) return {};

    const dates = klineData.map((d) => d.date);
    const ohlc = klineData.map((d) => [d.open, d.close, d.low, d.high]);
    const volumes = klineData.map((d) => d.volume);

    return {
      title: { text: `${klineCode} K线图`, left: 'center' },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
      },
      legend: { data: ['K线', '成交量'], bottom: 10 },
      grid: [
        { left: '10%', right: '8%', top: '10%', height: '50%' },
        { left: '10%', right: '8%', top: '70%', height: '15%' },
      ],
      xAxis: [
        { type: 'category', data: dates, gridIndex: 0 },
        { type: 'category', data: dates, gridIndex: 1, axisLabel: { show: false } },
      ],
      yAxis: [
        { scale: true, gridIndex: 0, splitLine: { show: true } },
        { scale: true, gridIndex: 1, splitLine: { show: false } },
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1], start: 50, end: 100 },
        { type: 'slider', xAxisIndex: [0, 1], bottom: 50 },
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: ohlc,
          itemStyle: {
            color: '#cf1322',
            color0: '#3f8600',
            borderColor: '#cf1322',
            borderColor0: '#3f8600',
          },
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes,
          itemStyle: {
            color: (params: any) => {
              const idx = params.dataIndex;
              return klineData[idx].close >= klineData[idx].open ? '#cf1322' : '#3f8600';
            },
          },
        },
      ],
    };
  };

  return (
    <div>
      {/* 数据状态概览 */}
      <Row gutter={16}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="股票总数"
              value={dataStatus?.total_stocks || 5234}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="数据源"
              value={dataStatus?.data_sources?.length || 2}
              prefix={<CloudServerOutlined />}
            />
            <div style={{ marginTop: 8 }}>
              {(dataStatus?.data_sources || ['tushare', 'akshare']).map((source: string) => (
                <Tag key={source} color="blue">{source}</Tag>
              ))}
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="更新状态"
              value={dataStatus?.update_status === 'idle' ? '正常' : '更新中'}
              prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
            />
            <div style={{ marginTop: 8, fontSize: 12, color: '#8c8c8c' }}>
              最后更新: {dataStatus?.last_update
                ? new Date(dataStatus.last_update).toLocaleString()
                : '2024-01-15 15:00:00'}
            </div>
          </Card>
        </Col>
      </Row>

      {/* K线图查看 */}
      <Card
        title="K线图查看"
        style={{ marginTop: 24 }}
        extra={
          <Space>
            <Input
              placeholder="输入股票代码"
              value={klineCode}
              onChange={(e) => setKlineCode(e.target.value)}
              style={{ width: 150 }}
              prefix={<SearchOutlined />}
            />
            <Button icon={<ReloadOutlined />} onClick={fetchKlineData}>
              刷新
            </Button>
          </Space>
        }
      >
        {klineLoading ? (
          <div style={{ textAlign: 'center', padding: 100 }}>
            <Spin size="large" />
          </div>
        ) : klineData && klineData.length > 0 ? (
          <ReactECharts option={getKlineOption()} style={{ height: 500 }} />
        ) : (
          <div style={{ textAlign: 'center', padding: 100, color: '#8c8c8c' }}>
            请输入股票代码查看K线图
          </div>
        )}
      </Card>
    </div>
  );
};

export default DataCenter;
