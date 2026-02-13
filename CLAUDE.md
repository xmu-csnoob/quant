# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A comprehensive A-Share (Chinese stock market) quantitative trading system built in Python. The system is designed to be platform-independent, supporting data acquisition, strategy development, backtesting, risk management, and live trading.

**Core Goal**: Implement quantitative strategies for stable returns in the Chinese A-share market (SSE, SZSE, BSE).

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set Tushare token (for real data)
export TUSHARE_TOKEN=your_token_here
```

## Common Commands

### Running Backtests
```bash
# Historical backtest
python apps/backtest/backtest_historical.py

# ML strategy backtest
python apps/backtest/backtest_ml_model.py
```

### Data Management
```bash
# Download historical data
python apps/data/download/batch_download.py

# Update latest data
python apps/data/update/data_update_service.py

# Quick test data module
python src/data/tests/quick_test.py
```

### Live Trading (Simulated/Paper)
```bash
# Daily routine (automated trading)
python apps/live/daily_routine.py

# Paper trading
python apps/live/live_paper_trading.py

# Check trading status
python apps/live/live_trading_status.py
```

### Monitoring
```bash
# Calculate PnL
python apps/monitor/calculate_pnl.py

# Check signals
python apps/monitor/check_signals.py
```

## Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Live Trading Layer                    │
│  LiveTradingEngine → TradingAPI → Broker/Platform       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                    Risk Management Layer                 │
│  RiskManager → PositionSizer → StopLoss/TakeProfit      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                      Strategy Layer                      │
│  Strategies → Signals → OrderManager                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                       Data Layer                         │
│  DataFetcher → Storage → Cache → DataManager            │
└─────────────────────────────────────────────────────────┘
```

### Key Directories

- `src/` - Core source code
  - `data/` - Data layer (fetchers, storage, cache, API)
  - `strategies/` - Trading strategies (6 strategies)
  - `backtesting/` - Backtesting engine
  - `risk/` - Risk management
  - `trading/` - Trading execution and order management
  - `utils/` - Utilities (indicators, features, ML)

- `apps/` - Application scripts
  - `data/` - Data download and update scripts
  - `backtest/` - Backtesting applications
  - `live/` - Live trading scripts
  - `monitor/` - Monitoring scripts

- `config/` - Configuration files
  - `settings.py` - Centralized configuration
  - `a_stock.yaml` - A-share specific rules

- `docs/` - Documentation
  - `SYSTEM_SUMMARY.md` - Complete system documentation
  - `designs/` - Architecture diagrams
  - `guides/` - User guides

## Core Components

### Data Layer

**DataManager** (`src/data/api/data_manager.py`) - Facade pattern for unified data access:
- Priority: Memory cache → Local file → API fetch
- Supports Mock, Tushare, and AkShare data sources

```python
from src.data.fetchers.mock import MockDataFetcher
from src.data.storage.storage import DataStorage
from src.data.api.data_manager import DataManager

fetcher = MockDataFetcher(scenario="bull")
manager = DataManager(fetcher=fetcher, storage=DataStorage())
df = manager.get_daily_price("600000.SH", "20230101", "20231231")
```

### Strategy Layer

**BaseStrategy** (`src/strategies/base.py`) - Abstract base class for all strategies:
- Strategies only generate signals, not responsible for backtesting or execution
- Implement `generate_signals(df) -> list[Signal]`

Available strategies:
- `MaMacdRsiStrategy` - Trend following (MA+MACD+RSI)
- `MeanReversionStrategy` - Mean reversion (Bollinger+RSI)
- `MLStrategy` - XGBoost-based prediction
- `EnsembleStrategy` - Voting/weighted combination
- `AdaptiveDynamicStrategy` - Market regime recognition

### Backtesting

**SimpleBacktester** (`src/backtesting/simple_backtester.py`):
- Fast backtesting for strategy validation
- Supports multiple strategy comparison
- Calculates win rate, max drawdown, Sharpe ratio

### Risk Management

**RiskManager** (`src/risk/manager.py`):
- Stop-loss/Take-profit (fixed ratio, trailing stop)
- Position sizing (fixed ratio, Kelly criterion, ATR-based)
- Drawdown control
- Consecutive loss protection

### Trading Execution

**LiveTradingEngine** (`src/trading/engine.py`):
- Subscribes to market data
- Executes strategy signals
- Risk checks
- Order management

## Configuration

### Central Configuration (`config/settings.py`)

Key parameters:
- Trading costs: Commission 0.03% buy, 0.13% sell (includes stamp duty)
- Position limits: Max 3 positions, 30% per stock
- Risk limits: 5% daily loss, 15% max drawdown, 10% stop loss
- Trading thresholds: Buy >52%, Sell <48%

### Environment Variables

```bash
export TUSHARE_TOKEN=your_token        # Tushare API token
export TUSHARE_ADVANCED_TOKEN=xxx      # Premium token (optional)
export DEBUG_MODE=true                  # Enable debug mode
export DRY_RUN=true                     # Simulation mode (no real trades)
```

## A-Share Trading Rules

- **Exchanges**: SSE (上交所), SZSE (深交所), BSE (北交所)
- **Trading hours**: 9:30-11:30, 13:00-15:00 (Beijing time)
- **T+1 rule**: Stocks bought today can only be sold tomorrow
- **Lot size**: Minimum 100 shares (1手)
- **Price limits**:
  - Main board: ±10%
  - ChiNext (创业板) / STAR Market (科创板): ±20%
  - BSE (北交所): ±30%
  - ST stocks: ±5%

## Git Submodules

The project is split into 4 repositories for parallel development:

| Repository | Responsibility |
|------------|----------------|
| quant-data | Data collection, storage, API |
| quant-strategies | Strategy research, backtesting, models |
| quant-trading | Live trading, risk control |
| quant-infra | Infrastructure, documentation |

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:xmu-csnoob/quant.git

# Update submodules
git submodule update --remote --merge
```

## Code Conventions

- **Docstrings**: Use Chinese for docstrings and comments
- **Type hints**: Encouraged for code clarity
- **Logging**: Use `loguru` for all logging
- **API tokens**: Never commit tokens; use environment variables
- **Default branch**: `main` (not `master`)

## Testing

Test coverage is minimal. Run existing tests:
```bash
python tests/strategies/test_strategy_backtest.py
```

## Recommended Workflow

1. **Development**: Use Mock data (fast, controllable)
2. **Validation**: Use Tushare real data
3. **Paper trading**: Use MockTradingAPI or GM platform
4. **Live trading**: Small capital first, then scale up

---

## Git Workflow & CI/CD

### Branch Protection Rules

The `main` branch is protected. **Direct pushes to `main` are NOT allowed.**

All changes must go through Pull Request (PR) with passing CI checks.

### Required CI Checks

Before merging, all PRs must pass these checks:
- `python-test (3.10)` - Python 3.10 tests
- `python-test (3.11)` - Python 3.11 tests
- `python-test (3.12)` - Python 3.12 tests
- `python-lint` - Code style checks (flake8, black, isort)
- `frontend` - Frontend build
- `security` - Dependency security scan

### Standard PR Workflow

```bash
# 1. Create a feature branch from main
git checkout main
git pull
git checkout -b feature/your-feature-name

# 2. Make your changes and commit
git add .
git commit -m "feat: your feature description"

# 3. Push the branch to remote
git push -u origin feature/your-feature-name

# 4. Create a Pull Request
gh pr create --title "feat: your feature description" --body "Description of changes"

# 5. Wait for CI to pass (check status)
gh pr checks <pr-number>

# 6. After CI passes and review, merge the PR
gh pr merge <pr-number> --merge --delete-branch

# 7. Sync local main
git checkout main
git pull
```

### Commit Message Convention

Use conventional commit format:

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code refactoring |
| `test:` | Adding/updating tests |
| `docs:` | Documentation changes |
| `ci:` | CI/CD changes |
| `chore:` | Maintenance tasks |

Example:
```
feat(trading): add price limit checking for buy/sell orders
```

### GitHub CLI (gh) Authentication

When working on a new machine, authenticate with:

```bash
# Login with token
echo "YOUR_GITHUB_TOKEN" | gh auth login --with-token

# Verify authentication
gh auth status
```

### Troubleshooting

**If push is rejected:**
```bash
# You're probably trying to push directly to main
# Create a branch instead:
git checkout -b feature/your-feature
git push -u origin feature/your-feature
```

**If CI fails:**
```bash
# Check what failed
gh pr checks <pr-number>

# View detailed logs
gh run view --log-failed
```

**Check CI status:**
```bash
gh run list --limit 5
```

