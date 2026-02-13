// 交易管理页面
import React, { useEffect, useState } from 'react';
import { Table, Card, Tabs, Tag, Button, Modal, Form, Input, InputNumber, Select, message, Space, Alert } from 'antd';
import {
  ReloadOutlined,
  PlusOutlined,
  LockOutlined,
} from '@ant-design/icons';
import dayjs from 'dayjs';
import { tradingApi, accountApi } from '../../api';
import type { Order, CreateOrderRequest, PaginatedResponse, Position } from '../../api/types';

const Trading: React.FC = () => {
  const [orders, setOrders] = useState<PaginatedResponse<Order> | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [activeTab, setActiveTab] = useState('all');

  useEffect(() => {
    fetchOrders();
    fetchPositions();
  }, [activeTab]);

  const fetchOrders = async () => {
    setLoading(true);
    try {
      const status = activeTab === 'all' ? undefined : activeTab;
      const data = await tradingApi.getOrders({ status, page: 1, page_size: 50 });
      setOrders(data);
    } catch (error) {
      console.error('获取订单失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPositions = async () => {
    try {
      const data = await accountApi.getPositions();
      setPositions(data);
    } catch (error) {
      console.error('获取持仓失败:', error);
    }
  };

  // 获取可卖数量
  const getAvailableShares = (code: string): number => {
    const pos = positions.find(p => p.code === code);
    return pos?.available || 0;
  };

  // 获取持仓信息
  const getPosition = (code: string): Position | undefined => {
    return positions.find(p => p.code === code);
  };

  const handleCreateOrder = async (values: CreateOrderRequest) => {
    // T+1检查：卖出时检查可卖数量
    if (values.direction === 'sell') {
      const available = getAvailableShares(values.code);
      if (values.shares > available) {
        const pos = getPosition(values.code);
        const locked = (pos?.shares || 0) - available;
        message.error(
          `可卖数量不足！总持仓${pos?.shares || 0}股，可卖${available}股，T+1锁定${locked}股`
        );
        return;
      }
    }

    try {
      await tradingApi.createOrder(values);
      message.success('订单创建成功');
      setModalVisible(false);
      form.resetFields();
      fetchOrders();
      fetchPositions();
    } catch (error: any) {
      // 显示后端返回的详细错误信息
      const errorMsg = error?.response?.data?.message || error?.message || '订单创建失败';
      message.error(errorMsg);
    }
  };

  const handleCancelOrder = async (orderId: string) => {
    try {
      await tradingApi.cancelOrder(orderId);
      message.success('订单已取消');
      fetchOrders();
    } catch (error) {
      message.error('取消订单失败');
    }
  };

  // 订单状态标签
  const getStatusTag = (status: string) => {
    const statusMap: Record<string, { color: string; text: string }> = {
      submitted: { color: 'processing', text: '已提交' },
      partial: { color: 'warning', text: '部分成交' },
      filled: { color: 'success', text: '全部成交' },
      cancelled: { color: 'default', text: '已撤单' },
      rejected: { color: 'error', text: '已拒绝' },
    };
    const config = statusMap[status] || { color: 'default', text: status };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 订单表格列定义
  const columns = [
    {
      title: '订单ID',
      dataIndex: 'order_id',
      key: 'order_id',
      width: 120,
      render: (id: string) => id.slice(0, 8) + '...',
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
      title: '类型',
      dataIndex: 'order_type',
      key: 'order_type',
      width: 80,
      render: (type: string) => type === 'limit' ? '限价' : '市价',
    },
    {
      title: '委托价',
      dataIndex: 'price',
      key: 'price',
      align: 'right' as const,
      width: 100,
      render: (price: number | null) => price ? `¥${price.toFixed(2)}` : '-',
    },
    {
      title: '委托数量',
      dataIndex: 'shares',
      key: 'shares',
      align: 'right' as const,
      width: 100,
      render: (shares: number) => shares.toLocaleString(),
    },
    {
      title: '成交数量',
      dataIndex: 'filled_shares',
      key: 'filled_shares',
      align: 'right' as const,
      width: 100,
      render: (filled: number, record: Order) => (
        <span>
          {filled.toLocaleString()}
          {filled > 0 && filled < record.shares && (
            <span style={{ color: '#8c8c8c', fontSize: 12 }}> / {record.shares.toLocaleString()}</span>
          )}
        </span>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => getStatusTag(status),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (time: string) => dayjs(time).format('YYYY-MM-DD HH:mm:ss'),
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (_: any, record: Order) => (
        record.status === 'submitted' && (
          <Button
            type="link"
            danger
            size="small"
            onClick={() => handleCancelOrder(record.order_id)}
          >
            撤单
          </Button>
        )
      ),
    },
  ];

  return (
    <div>
      <Card
        title="交易管理"
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={fetchOrders}>
              刷新
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
              新建订单
            </Button>
          </Space>
        }
      >
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            { key: 'all', label: '全部订单' },
            { key: 'submitted', label: '待成交' },
            { key: 'filled', label: '已成交' },
            { key: 'cancelled', label: '已撤单' },
          ]}
        />

        <Table
          columns={columns}
          dataSource={orders?.items || []}
          rowKey="order_id"
          loading={loading}
          pagination={{
            total: orders?.total || 0,
            pageSize: orders?.page_size || 20,
            current: orders?.page || 1,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
          }}
        />
      </Card>

      {/* 新建订单弹窗 */}
      <Modal
        title="新建订单"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        onOk={() => form.submit()}
        okText="提交"
        cancelText="取消"
      >
        <Alert
          message="A股交易规则"
          description={
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              <li>T+1规则：当日买入的股票，下一交易日才能卖出</li>
              <li>最小交易单位：100股（1手）</li>
              <li>涨跌停限制：主板±10%，创业板/科创板±20%</li>
            </ul>
          }
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateOrder}
          initialValues={{ order_type: 'limit', shares: 100 }}
        >
          <Form.Item
            name="code"
            label="股票代码"
            rules={[{ required: true, message: '请输入股票代码' }]}
          >
            <Input placeholder="例如: 600000.SH" onChange={() => form.validateFields()} />
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prev, curr) => prev.code !== curr.code}
          >
            {({ getFieldValue }) => {
              const code = getFieldValue('code');
              const pos = getPosition(code);
              if (pos) {
                return (
                  <Alert
                    message={`持仓: ${pos.shares}股 | 可卖: ${pos.available}股 | 成本: ¥${pos.cost_price}`}
                    type={pos.available > 0 ? 'success' : 'warning'}
                    style={{ marginBottom: 16 }}
                    icon={pos.available < pos.shares ? <LockOutlined /> : undefined}
                  />
                );
              }
              return null;
            }}
          </Form.Item>

          <Form.Item
            name="direction"
            label="买卖方向"
            rules={[{ required: true, message: '请选择买卖方向' }]}
          >
            <Select>
              <Select.Option value="buy">
                <span style={{ color: '#cf1322' }}>买入</span>
              </Select.Option>
              <Select.Option value="sell">
                <span style={{ color: '#3f8600' }}>卖出</span>
              </Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prev, curr) => prev.direction !== curr.direction || prev.code !== curr.code}
          >
            {({ getFieldValue }) => {
              const direction = getFieldValue('direction');
              const code = getFieldValue('code');
              if (direction === 'sell' && code) {
                const available = getAvailableShares(code);
                if (available === 0) {
                  return (
                    <Alert
                      message="无可卖持仓"
                      description="该股票无可卖数量（可能为T+1锁定或未持仓）"
                      type="error"
                      style={{ marginBottom: 16 }}
                    />
                  );
                }
              }
              return null;
            }}
          </Form.Item>

          <Form.Item
            name="order_type"
            label="订单类型"
            rules={[{ required: true, message: '请选择订单类型' }]}
          >
            <Select>
              <Select.Option value="limit">限价单</Select.Option>
              <Select.Option value="market">市价单</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prev, curr) => prev.order_type !== curr.order_type}
          >
            {({ getFieldValue }) =>
              getFieldValue('order_type') === 'limit' && (
                <Form.Item
                  name="price"
                  label="委托价格"
                  rules={[{ required: true, message: '请输入委托价格' }]}
                >
                  <InputNumber
                    style={{ width: '100%' }}
                    min={0.01}
                    step={0.01}
                    precision={2}
                    prefix="¥"
                  />
                </Form.Item>
              )
            }
          </Form.Item>

          <Form.Item
            name="shares"
            label="委托数量"
            rules={[{ required: true, message: '请输入委托数量' }]}
            extra="最小100股，必须是100的整数倍"
          >
            <InputNumber
              style={{ width: '100%' }}
              min={100}
              step={100}
              precision={0}
              suffix="股"
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default Trading;
