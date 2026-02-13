import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './styles/index.css'

// Ant Design 配置 - 金融主题
const themeConfig = {
  token: {
    colorPrimary: '#1890ff',
    colorSuccess: '#3f8600',
    colorWarning: '#faad14',
    colorError: '#cf1322',
    borderRadius: 6,
  },
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider theme={themeConfig} locale={zhCN}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ConfigProvider>
  </StrictMode>,
)
