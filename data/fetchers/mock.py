"""
模拟数据获取器

用于开发阶段和测试，生成符合 A 股特征的模拟 OHLCV 数据
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from loguru import logger

from data.fetchers.base import (
    BaseDataFetcher,
    Exchange,
    DataFetchError
)


class MockDataFetcher(BaseDataFetcher):
    """
    模拟数据获取器

    特点：
    1. 不需要网络和 API Token
    2. 数据可控、可重复（固定随机种子）
    3. 支持多种市场场景（牛市、熊市、横盘等）
    4. 快速生成大量测试数据

    用途：
    - 开发阶段快速验证功能
    - 单元测试（确定性数据）
    - 策略回测（可重复的测试场景）
    """

    # 支持的市场场景
    SCENARIOS = [
        "normal",      # 正常市场（随机游走）
        "bull",        # 牛市（持续上涨）
        "bear",        # 熊市（持续下跌）
        "sideways",    # 横盘震荡
        "volatile",    # 高波动
        "gap_up",      # 跳空上涨
        "gap_down",    # 跳空下跌
        "limit_up",    # 涨停板
        "limit_down",  # 跌停板
    ]

    # 模拟股票列表
    MOCK_STOCKS = {
        Exchange.SSE: [
            "600000.SH", "600036.SH", "600519.SH",
            "600887.SH", "601318.SH", "601398.SH",
            "601857.SH", "601939.SH", "603259.SH", "688981.SH"
        ],
        Exchange.SZSE: [
            "000001.SZ", "000002.SZ", "000333.SZ",
            "000858.SZ", "002415.SZ", "300014.SZ",
            "300059.SZ", "300142.SZ", "300750.SZ", "301001.SZ"
        ],
        Exchange.BSE: [
            "836079.BJ", "832566.BJ", "830799.BJ"
        ]
    }

    # 股票名称映射
    STOCK_NAMES = {
        "600000.SH": "浦发银行",
        "600036.SH": "招商银行",
        "600519.SH": "贵州茅台",
        "600887.SH": "伊利股份",
        "601318.SH": "中国平安",
        "601398.SH": "工商银行",
        "601857.SH": "中国石油",
        "601939.SH": "建设银行",
        "603259.SH": "药明康德",
        "688981.SH": "中芯国际",
        "000001.SZ": "平安银行",
        "000002.SZ": "万科A",
        "000333.SZ": "美的集团",
        "000858.SZ": "五粮液",
        "002415.SZ": "海康威视",
        "300014.SZ": "亿纬锂能",
        "300059.SZ": "东方财富",
        "300142.SZ": "沃森生物",
        "300750.SZ": "宁德时代",
        "301001.SZ": "凯盛新材",
        "836079.BJ": "安达科技",
        "832566.BJ": "梓撞科技",
        "830799.BJ": "华燕照明",
    }

    def __init__(
        self,
        scenario: str = "normal",
        mock_data_dir: str = "data/tests/fixtures/mock_data"
    ):
        """
        初始化模拟数据获取器

        Args:
            scenario: 市场场景类型（见 SCENARIOS）
            mock_data_dir: 模拟数据保存目录
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Available scenarios: {self.SCENARIOS}"
            )

        self.scenario = scenario
        self.mock_data_dir = Path(mock_data_dir)
        self.mock_data_dir.mkdir(parents=True, exist_ok=True)

        # 预生成的模拟数据缓存
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"MockDataFetcher initialized with scenario: {scenario}")

    def get_stock_list(self, exchange: Exchange) -> pd.DataFrame:
        """
        获取模拟股票列表

        返回固定的测试股票列表

        Args:
            exchange: 交易所枚举

        Returns:
            股票列表 DataFrame
        """
        stocks = self.MOCK_STOCKS.get(exchange, [])

        if not stocks:
            logger.warning(f"No mock stocks for exchange: {exchange}")
            return pd.DataFrame()

        # 构造 DataFrame
        df = pd.DataFrame({
            "ts_code": stocks,
            "symbol": [s.split(".")[0] for s in stocks],
            "name": [self.STOCK_NAMES.get(s, f"测试股票{i}") for i, s in enumerate(stocks)],
            "area": ["北京"] * len(stocks),
            "industry": ["金融"] * len(stocks),
            "market": ["主板"] * len(stocks),
            "list_date": ["20200101"] * len(stocks)
        })

        logger.debug(f"Returned {len(df)} mock stocks for {exchange.value}")

        return df

    def get_daily_price(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取模拟日线数据

        根据场景类型生成不同的数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            日线数据 DataFrame
        """
        # 检查缓存
        cache_key = f"{ts_code}_{start_date}_{end_date}_{self.scenario}"
        if cache_key in self._cache:
            df = self._cache[cache_key].copy()
            logger.debug(f"Cache hit for {ts_code} ({self.scenario})")
            return df

        # 生成模拟数据
        try:
            df = self._generate_mock_data(
                ts_code,
                start_date,
                end_date,
                self.scenario
            )

            # 缓存
            self._cache[cache_key] = df.copy()

            logger.debug(
                f"Generated {len(df)} mock records for {ts_code} "
                f"({start_date} ~ {end_date}, scenario={self.scenario})"
            )

            return df

        except Exception as e:
            logger.error(f"Failed to generate mock data for {ts_code}: {e}")
            raise DataFetchError(f"Mock data generation failed: {e}")

    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取模拟指数日线数据

        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指数数据 DataFrame
        """
        # 指数数据生成逻辑与股票类似
        cache_key = f"{index_code}_{start_date}_{end_date}_{self.scenario}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        df = self._generate_mock_data(
            index_code,
            start_date,
            end_date,
            self.scenario
        )

        self._cache[cache_key] = df.copy()

        return df

    def set_scenario(self, scenario: str) -> None:
        """
        切换市场场景

        Args:
            scenario: 场景类型（见 SCENARIOS）

        Raises:
            ValueError: 未知的场景类型
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario}. "
                f"Available: {self.SCENARIOS}"
            )

        self.scenario = scenario
        # 清空缓存
        self._cache.clear()

        logger.info(f"Mock scenario switched to: {scenario}")

    def _generate_mock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        scenario: str
    ) -> pd.DataFrame:
        """
        生成模拟 OHLCV 数据

        根据 A 股特征生成：
        - T+1 交易制度
        - 涨跌停限制（主板 ±10%, 科创板/创业板 ±20%, 北交所 ±30%）
        - 交易时间（周一至周五）
        - 最小变动价位（0.01 元）
        - 成交量单位：手（1 手 = 100 股）

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            scenario: 市场场景

        Returns:
            模拟数据 DataFrame
        """
        # 计算交易日（排除周末）
        dates = pd.date_range(start_date, end_date, freq="B")
        dates = [d for d in dates if d.weekday() < 5]

        if len(dates) == 0:
            return pd.DataFrame()

        # 初始价格（根据股票代码生成，保证每只股票不同）
        initial_price = 10.0 + hash(ts_code) % 90  # 10-100 元

        # 根据场景生成价格序列
        if scenario == "normal":
            prices = self._generate_normal_prices(len(dates), initial_price)
        elif scenario == "bull":
            prices = self._generate_bull_market_prices(len(dates), initial_price)
        elif scenario == "bear":
            prices = self._generate_bear_market_prices(len(dates), initial_price)
        elif scenario == "sideways":
            prices = self._generate_sideways_prices(len(dates), initial_price)
        elif scenario == "volatile":
            prices = self._generate_volatile_prices(len(dates), initial_price)
        else:
            # 默认使用正常市场
            prices = self._generate_normal_prices(len(dates), initial_price)

        # 生成 OHLCV
        n_days = len(dates)
        n_prices = len(prices["open"])  # OHLC 的长度（可能比 dates 少 1）

        df = pd.DataFrame({
            "ts_code": [ts_code] * n_prices,
            "trade_date": [d.strftime("%Y%m%d") for d in dates[:n_prices]],
            "open": prices["open"],
            "high": prices["high"],
            "low": prices["low"],
            "close": prices["close"],
            "volume": np.random.randint(100000, 10000000, n_prices),  # 成交量（手）
            "amount": [  # 成交额（千元）= 成交量 * 收盘价 / 100
                c * v / 10000
                for c, v in zip(prices["close"], np.random.randint(100000, 10000000, n_prices))
            ]
        })

        return df

    def _generate_normal_prices(
        self,
        n_days: int,
        initial_price: float
    ) -> dict:
        """
        生成正常市场的价格（随机游走）

        模拟：涨跌概率各 50%，日波动 ±3%
        """
        # 使用股票代码作为随机种子的一部分，保证同一股票每次生成相同数据
        seed = 42
        np.random.seed(seed)

        # 日收益率：均值 0，标准差 2%
        returns = np.random.normal(0, 0.02, n_days)

        # 生成价格序列
        prices = [initial_price]
        for ret in returns[1:]:
            price = prices[-1] * (1 + ret)
            prices.append(max(price, 0.01))  # 价格不能小于 0.01

        # 生成 OHLC
        return self._prices_to_ohlc(prices)

    def _generate_bull_market_prices(
        self,
        n_days: int,
        initial_price: float
    ) -> dict:
        """
        生成牛市价格（持续上涨）

        模拟：70% 概率上涨，平均日涨幅 1%
        """
        np.random.seed(42)

        # 日收益率：均值 1%，标准差 3%
        returns = np.random.normal(0.01, 0.03, n_days)

        prices = [initial_price]
        for ret in returns[1:]:
            price = prices[-1] * (1 + ret)
            prices.append(max(price, 0.01))

        return self._prices_to_ohlc(prices)

    def _generate_bear_market_prices(
        self,
        n_days: int,
        initial_price: float
    ) -> dict:
        """
        生成熊市价格（持续下跌）

        模拟：70% 概率下跌，平均日跌幅 -1%
        """
        np.random.seed(42)

        # 日收益率：均值 -1%，标准差 3%
        returns = np.random.normal(-0.01, 0.03, n_days)

        prices = [initial_price]
        for ret in returns[1:]:
            price = prices[-1] * (1 + ret)
            prices.append(max(price, 0.01))

        return self._prices_to_ohlc(prices)

    def _generate_sideways_prices(
        self,
        n_days: int,
        initial_price: float
    ) -> dict:
        """
        生成横盘价格（震荡）

        模拟：价格在 ±5% 范围内波动
        """
        np.random.seed(42)

        # 日收益率：均值 0，标准差 1%（小波动）
        returns = np.random.normal(0, 0.01, n_days)

        prices = [initial_price]
        for ret in returns[1:]:
            price = prices[-1] * (1 + ret)
            # 限制在初始价格的 ±5%
            price = max(min(price, initial_price * 1.05), initial_price * 0.95)
            prices.append(price)

        return self._prices_to_ohlc(prices)

    def _generate_volatile_prices(
        self,
        n_days: int,
        initial_price: float
    ) -> dict:
        """
        生成高波动价格

        模拟：日波动 ±5%
        """
        np.random.seed(42)

        # 日收益率：均值 0，标准差 5%（大波动）
        returns = np.random.normal(0, 0.05, n_days)

        prices = [initial_price]
        for ret in returns[1:]:
            price = prices[-1] * (1 + ret)
            prices.append(max(price, 0.01))

        return self._prices_to_ohlc(prices)

    @staticmethod
    def _prices_to_ohlc(prices: list) -> dict:
        """
        将价格序列转换为 OHLC

        Args:
            prices: 收盘价序列

        Returns:
            包含 open, high, low, close 的字典
        """
        if len(prices) < 2:
            return {
                "open": [],
                "high": [],
                "low": [],
                "close": []
            }

        opens = prices[:-1]
        closes = prices[1:]

        # 生成 high 和 low（基于 open 和 close）
        np.random.seed(42)

        highs = [
            max(o, c) * (1 + abs(np.random.normal(0, 0.01)))
            for o, c in zip(opens, closes)
        ]
        lows = [
            min(o, c) * (1 - abs(np.random.normal(0, 0.01)))
            for o, c in zip(opens, closes)
        ]

        return {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.debug("Mock data cache cleared")
