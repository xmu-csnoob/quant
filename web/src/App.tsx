import { Routes, Route } from 'react-router-dom'
import MainLayout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Trading from './pages/Trading'
import StrategyCenter from './pages/Strategy'
import Backtest from './pages/Backtest'
import RiskManagement from './pages/Risk'
import DataCenter from './pages/Data'
import MLPrediction from './pages/ML'

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="trading" element={<Trading />} />
        <Route path="strategy" element={<StrategyCenter />} />
        <Route path="backtest" element={<Backtest />} />
        <Route path="risk" element={<RiskManagement />} />
        <Route path="data" element={<DataCenter />} />
        <Route path="ml" element={<MLPrediction />} />
      </Route>
    </Routes>
  )
}

export default App
