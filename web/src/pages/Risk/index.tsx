// 风险管理页面
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Form, InputNumber, Slider, Switch, Button, Statistic, Alert, Divider, message } from 'antd';
import { SaveOutlined, ReloadOutlined } from '@ant-design/icons';
import { riskApi } from '../../api';
import type { RiskStatus } from '../../api/types';

const RiskManagement: React.FC = () => {
  const [form] = Form.useForm();
  const [riskStatus, setRiskStatus] = useState<RiskStatus | null>(null);
  const [saving, setSaving] = useState(false);

  // 加载配置
  const fetchConfig = async () => {
    try {
      const data = await riskApi.getConfig();
      form.setFieldsValue(data);
    } catch (error) {
      console.error('获取风控配置失败:', error);
      message.error('获取配置失败');
    }
  };

  // 加载风险状态
  const fetchRiskStatus = async () => {
    try {
      const status = await riskApi.getStatus();
      setRiskStatus(status);
    } catch (error) {
      console.error('获取风控状态失败:', error);
      message.error('获取状态失败');
    }
  };

  useEffect(() => {
    fetchConfig();
    fetchRiskStatus();
  }, []);

  const handleSave = async (values: any) => {
    setSaving(true);
    try {
      await riskApi.saveConfig(values);
      message.success('风控配置已保存');
      await fetchConfig();
      await fetchRiskStatus();
    } catch (error: any) {
      message.error('保存失败: ' + (error?.message || '未知错误'));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <Row gutter={24}>
        {/* 左侧：风控配置 */}
        <Col xs={24} lg={12}>
          <Card title="风控参数配置">
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                max_position_count: 3,
                max_position_ratio: 30,
                min_position_ratio: 5,
                stop_loss_ratio: 10,
                take_profit_ratio: 20,
                max_daily_loss: 5,
                max_drawdown: 15,
                enable_auto_stop_loss: true,
                enable_trailing_stop: false,
                trailing_stop_ratio: 5,
                enable_consecutive_loss: true,
                max_consecutive_losses: 3,
                enable_t1_rule: true,
              }}
              onFinish={handleSave}
            >
              <Divider>仓位限制</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="最大持仓数量" name="max_position_count">
                    <InputNumber min={1} max={10} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="单股最大仓位(%)" name="max_position_ratio">
                    <Slider min={5} max={50} marks={{ 5: '5%', 25: '25%', 50: '50%' }} />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>止损止盈</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="止损比例(%)" name="stop_loss_ratio">
                    <InputNumber
                      min={1}
                      max={50}
                      style={{ width: '100%' }}
                      addonAfter="%"
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="止盈比例(%)" name="take_profit_ratio">
                    <InputNumber
                      min={1}
                      max={100}
                      style={{ width: '100%' }}
                      addonAfter="%"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>自动止损</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="启用自动止损" name="enable_auto_stop_loss" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="启用移动止损" name="enable_trailing_stop" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="移动止损回撤(%)" name="trailing_stop_ratio">
                    <InputNumber
                      min={0}
                      max={20}
                      style={{ width: '100%' }}
                      addonAfter="%"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>风控阈值</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="单日最大亏损(%)" name="max_daily_loss">
                    <InputNumber
                      min={1}
                      max={20}
                      style={{ width: '100%' }}
                      addonAfter="%"
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="最大回撤(%)" name="max_drawdown">
                    <InputNumber
                      min={1}
                      max={50}
                      style={{ width: '100%' }}
                      addonAfter="%"
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>连续亏损保护</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="启用连续亏损保护" name="enable_consecutive_loss" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="最大连续亏损次数" name="max_consecutive_losses">
                    <InputNumber
                      min={1}
                      max={10}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>T+1规则</Divider>
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item label="启用T+1规则" name="enable_t1_rule" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>
            </Form>

            <div style={{ marginTop: 24, textAlign: 'center' }}>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SaveOutlined />}
                loading={saving}
                onClick={() => form.submit()}
              >
                保存配置
              </Button>
            </div>
          </Card>
        </Col>

        {/* 右侧：风险监控 */}
        <Col xs={24} lg={12}>
          <Card
            title="风险监控"
            extra={
              <Button icon={<ReloadOutlined />} onClick={() => { fetchConfig(); fetchRiskStatus(); }}>
                刷新
              </Button>
            }
          >
            {/* 风控状态 */}
            <Card style={{ marginBottom: 16 }}>
              <Alert
                message={riskStatus?.risk_level === 'high' ? '高风险警告' : '风控状态正常'}
                description="当前未触发任何风控规则"
                type={riskStatus?.risk_level === 'high' ? 'error' : 'success'}
                showIcon
                style={{ marginBottom: 16 }}
              />
              <Row gutter={[16, 16]}>
                <Col xs={12} sm={8} md={6}>
                  <Statistic title="持仓数量" value={riskStatus?.current_positions || 0} />
                </Col>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="仓位比例"
                      value={riskStatus?.position_ratio || 0}
                      suffix="%"
                      valueStyle={{ color: (riskStatus?.position_ratio || 0) > 30 ? '#cf1322' : '#3f8600' }}
                    />
                  </Card>
                </Col>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="当日亏损"
                      value={riskStatus?.daily_loss || 0}
                      suffix="元"
                      valueStyle={{ color: (riskStatus?.daily_loss || 0) > 0 ? '#cf1322' : '#3f8600' }}
                    />
                  </Card>
                </Col>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="今日亏损率"
                      value={riskStatus?.daily_loss_ratio || 0}
                      suffix="%"
                      valueStyle={{ color: (riskStatus?.daily_loss_ratio || 0) > 0 ? '#cf1322' : '#3f8600' }}
                    />
                  </Card>
                </Col>
              </Row>
              <Row gutter={[16, 16]}>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="最大回撤"
                      value={riskStatus?.max_drawdown || 0}
                      suffix="%"
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Card>
                </Col>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="当前回撤"
                      value={(riskStatus?.max_drawdown_ratio || 0).toFixed(1)}
                      suffix="%"
                      valueStyle={{ color: (riskStatus?.max_drawdown_ratio || 0) > 10 ? '#cf1322' : '#52c41a' }}
                    />
                  </Card>
                </Col>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="连续亏损"
                      value={riskStatus?.consecutive_losses || 0}
                      suffix="次"
                      valueStyle={{ color: (riskStatus?.consecutive_losses || 0) >= 3 ? '#cf1322' : '#52c41a' }}
                    />
                  </Card>
                </Col>
                <Col xs={12} sm={8} md={6}>
                  <Card size="small">
                    <Statistic
                      title="T+1锁定"
                      value={riskStatus?.t1_locked_shares || 0}
                      suffix="股"
                      valueStyle={{ color: '#faad14' }}
                    />
                  </Card>
                </Col>
              </Row>

              <Divider />

              {/* 风控事件日志 */}
              <Card
                title="风控事件日志"
                style={{ marginTop: 16 }}
              >
                <div style={{ maxHeight: 200, overflow: 'auto' }}>
                  <p style={{ color: '#8c8c8c', textAlign: 'center', padding: 50 }}>暂无风控事件</p>
                </div>
              </Card>
            </Card>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default RiskManagement;
