// 主布局组件
import React, { useState } from 'react';
import { Layout, Menu, Avatar, Dropdown, Badge, Button, theme } from 'antd';
import {
  DashboardOutlined,
  StockOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  SafetyOutlined,
  DatabaseOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  BellOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useUIStore } from '../../stores';
import './index.css';

const { Header, Sider, Content } = Layout;

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '仪表盘' },
  { key: '/trading', icon: <StockOutlined />, label: '交易管理' },
  { key: '/strategy', icon: <ThunderboltOutlined />, label: '策略中心' },
  { key: '/ml', icon: <RobotOutlined />, label: 'ML预测' },
  { key: '/backtest', icon: <LineChartOutlined />, label: '回测分析' },
  { key: '/risk', icon: <SafetyOutlined />, label: '风险管理' },
  { key: '/data', icon: <DatabaseOutlined />, label: '数据中心' },
];

const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { collapsed, toggleCollapsed } = useUIStore();
  const [notifications] = useState(3);
  const { token: { colorBgContainer, borderRadiusLG } } = theme.useToken();

  const userMenuItems = [
    { key: 'settings', icon: <SettingOutlined />, label: '系统设置' },
    { key: 'logout', icon: <LogoutOutlined />, label: '退出登录' },
  ];

  const handleMenuClick = (key: string) => {
    navigate(key);
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* 侧边栏 */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        theme="dark"
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
        }}
      >
        {/* Logo */}
        <div className="logo">
          <span className="logo-text">{collapsed ? '量' : 'A股量化系统'}</span>
        </div>

        {/* 导航菜单 */}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => handleMenuClick(key)}
        />
      </Sider>

      {/* 右侧内容区 */}
      <Layout style={{ marginLeft: collapsed ? 80 : 200, transition: 'margin-left 0.2s' }}>
        {/* 顶部栏 */}
        <Header
          style={{
            padding: '0 24px',
            background: colorBgContainer,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
            position: 'sticky',
            top: 0,
            zIndex: 10,
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={toggleCollapsed}
            style={{ fontSize: '16px', width: 48, height: 48 }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {/* 通知 */}
            <Badge count={notifications} size="small">
              <Button type="text" icon={<BellOutlined />} style={{ fontSize: 18 }} />
            </Badge>

            {/* 用户菜单 */}
            <Dropdown
              menu={{
                items: userMenuItems,
                onClick: ({ key }) => {
                  if (key === 'logout') {
                    // 退出登录逻辑
                    console.log('logout');
                  }
                },
              }}
              placement="bottomRight"
            >
              <div style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Avatar size="small" icon={<UserOutlined />} />
                <span>量化交易员</span>
              </div>
            </Dropdown>
          </div>
        </Header>

        {/* 内容区 */}
        <Content
          style={{
            margin: 24,
            padding: 24,
            minHeight: 280,
            background: colorBgContainer,
            borderRadius: borderRadiusLG,
          }}
        >
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;
