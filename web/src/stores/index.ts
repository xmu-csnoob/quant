// Zustand Store - 全局状态管理
import { create } from 'zustand';
import type { AccountSummary, Position, Signal, Strategy } from '../api/types';

// 账户状态
interface AccountState {
  summary: AccountSummary | null;
  positions: Position[];
  setSummary: (summary: AccountSummary) => void;
  setPositions: (positions: Position[]) => void;
}

export const useAccountStore = create<AccountState>((set) => ({
  summary: null,
  positions: [],
  setSummary: (summary) => set({ summary }),
  setPositions: (positions) => set({ positions }),
}));

// 策略状态
interface StrategyState {
  strategies: Strategy[];
  signals: Signal[];
  setStrategies: (strategies: Strategy[]) => void;
  setSignals: (signals: Signal[]) => void;
  addSignal: (signal: Signal) => void;
}

export const useStrategyStore = create<StrategyState>((set) => ({
  strategies: [],
  signals: [],
  setStrategies: (strategies) => set({ strategies }),
  setSignals: (signals) => set({ signals }),
  addSignal: (signal) => set((state) => ({ signals: [signal, ...state.signals].slice(0, 100) })),
}));

// UI状态
interface UIState {
  collapsed: boolean;
  theme: 'light' | 'dark';
  toggleCollapsed: () => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useUIStore = create<UIState>((set) => ({
  collapsed: false,
  theme: 'light',
  toggleCollapsed: () => set((state) => ({ collapsed: !state.collapsed })),
  setTheme: (theme) => set({ theme }),
}));
