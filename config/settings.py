"""
量化交易系统 - 统一配置管理

所有配置参数集中管理，避免分散在多个文件中
"""

from pathlib import Path
from decimal import Decimal
import os

# ==================== 项目路径 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# ==================== A股交易参数 ====================
# 交易成本（符合A股规则）
BUY_COST_RATE = Decimal('0.0003')    # 买入: 0.03% 佣金
SELL_COST_RATE = Decimal('0.0013')   # 卖出: 0.13% 佣金+印花税

# 交易规则
MIN_LOT_SIZE = 100                   # 最小交易单位（手）
TRADING_HOURS = {
    "morning": ("09:30", "11:30"),
    "afternoon": ("13:00", "15:00")
}

# ==================== 策略参数 ====================
MAX_POSITIONS = 3                    # 最大持仓数量
POSITION_SIZE = Decimal('0.3')       # 单只股票仓位比例（30%）
BUY_THRESHOLD = Decimal('0.52')      # 买入阈值（上涨概率>52%）
SELL_THRESHOLD = Decimal('0.48')     # 卖出阈值（上涨概率<48%）
REBALANCE_THRESHOLD = Decimal('0.05') # 调仓阈值（新信号优势>5%）

# 股票池配置
UNIVERSE_SIZE = 500                  # 扫描股票数量
MIN_HISTORY_DAYS = 60                # 最小历史数据天数
FEATURE_LOOKBACK_DAYS = 120          # 特征计算回看天数

# ==================== 数据源配置 ====================
# Tushare配置
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')
TUSHARE_API_URL = "http://api.tushare.pro"

# 高级Token配置（可选）
ADVANCED_TOKEN = os.getenv('TUSHARE_ADVANCED_TOKEN', '')
PROXY_URL = os.getenv('TUSHARE_PROXY_URL', '')

# 数据更新策略
DATA_FETCH_DELAY = 0.2               # 请求间隔（秒）
MAX_RETRIES = 3                      # 最大重试次数

# ==================== 模型配置 ====================
CURRENT_MODEL = MODELS_DIR / "xgboost_2022_2026.json"
MODEL_RETRAIN_INTERVAL_DAYS = 30     # 模型重训练间隔（天）

# ==================== 定时任务配置 ====================
# 每日交易例程执行时间
DAILY_ROUTINE_HOUR = 19              # 19:00执行（确保数据已更新）
DAILY_ROUTINE_MINUTE = 0
DAILY_ROUTINE_WEEKDAYS = range(1, 6) # 周一到周五

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
LOG_ROTATION = "1 day"
LOG_RETENTION = "30 days"

# ==================== 状态文件配置 ====================
STATE_FILE = DATA_DIR / "live_trading_state.json"
STATE_BACKUP_DIR = DATA_DIR / "state_backups"
MAX_BACKUPS = 7                      # 保留7天备份

# ==================== 健康检查配置 ====================
HEARTBEAT_FILE = LOGS_DIR / "heartbeat.log"
HEALTH_CHECK_INTERVAL_MINUTES = 60   # 健康检查间隔

# ==================== 风险管理配置 ====================
MAX_DAILY_LOSS_PCT = Decimal('0.05') # 单日最大亏损5%
MAX_DRAWDOWN_PCT = Decimal('0.15')   # 最大回撤15%
STOP_LOSS_PCT = Decimal('0.10')      # 单股止损10%

# ==================== 通知配置（可选）====================
# 钉钉通知
DINGTALK_WEBHOOK = os.getenv('DINGTALK_WEBHOOK', '')
DINGTALK_SECRET = os.getenv('DINGTALK_SECRET', '')

# 邮件通知
EMAIL_SMTP_HOST = os.getenv('EMAIL_SMTP_HOST', '')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
EMAIL_USER = os.getenv('EMAIL_USER', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO = os.getenv('EMAIL_TO', '').split(',')

# ==================== 调试配置 ====================
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'  # 模拟运行，不实际交易


def get_state_file_path() -> Path:
    """获取状态文件路径"""
    return STATE_FILE


def get_model_path() -> Path:
    """获取模型文件路径"""
    return CURRENT_MODEL


def get_log_file(name: str = "daily_trading") -> Path:
    """获取日志文件路径"""
    return LOGS_DIR / f"{name}.log"


def is_trading_day() -> bool:
    """
    判断今天是否为交易日

    TODO: 实现完整的交易日历检查
    目前简单判断：周一到周五且不是法定节假日
    """
    from datetime import datetime
    import holidays

    cn_holidays = holidays.CountryHoliday('CN')

    now = datetime.now()
    # 检查是否为周一到周五
    if now.weekday() not in DAILY_ROUTINE_WEEKDAYS:
        return False

    # 检查是否为法定节假日
    if now.date() in cn_holidays:
        return False

    return True


def validate_config() -> dict:
    """
    验证配置有效性

    Returns:
        dict: 验证结果 {'valid': bool, 'errors': list}
    """
    errors = []

    # 检查必需的目录
    for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR]:
        if not dir_path.exists():
            errors.append(f"目录不存在: {dir_path}")

    # 检查模型文件
    if not CURRENT_MODEL.exists():
        errors.append(f"模型文件不存在: {CURRENT_MODEL}")

    # 检查状态文件
    if not STATE_FILE.exists():
        errors.append(f"状态文件不存在: {STATE_FILE}")

    # 检查Tushare Token
    if not TUSHARE_TOKEN and not ADVANCED_TOKEN:
        errors.append("未设置TUSHARE_TOKEN或TUSHARE_ADVANCED_TOKEN")

    # 验证参数范围
    if BUY_THRESHOLD <= SELL_THRESHOLD:
        errors.append(f"买入阈值({BUY_THRESHOLD})必须大于卖出阈值({SELL_THRESHOLD})")

    if MAX_POSITIONS < 1:
        errors.append(f"最大持仓数({MAX_POSITIONS})必须>=1")

    if POSITION_SIZE * MAX_POSITIONS > Decimal('1.0'):
        errors.append(f"总仓位({POSITION_SIZE * MAX_POSITIONS})不能超过100%")

    return {
        'valid': len(errors) == 0,
        'errors': errors
    }


# ==================== 辅助函数 ====================

def to_decimal(value) -> Decimal:
    """将数值转换为Decimal，确保货币计算精度"""
    return Decimal(str(value))


def format_money(value) -> str:
    """格式化货币显示"""
    if isinstance(value, Decimal):
        return f"{value:,.2f}"
    return f"{float(value):,.2f}"
