import os

BASE_DIR = os.path.dirname(__file__)


def _load_env_file(path: str):
    """
    Loads KEY=value pairs from a local .env file without overriding
    variables already set in the shell environment.
    """
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(os.path.join(BASE_DIR, ".env"))

# Top 20 Manual Symbol Universe (Binance Futures)
SYMBOLS = [
    "btcusdt",
    "ethusdt",
    "bnbusdt",
    "solusdt",
    "xrpusdt",
    "dogeusdt",
    "adausdt",
    "trxusdt",
    "linkusdt",
    "avaxusdt",
    "suiusdt",
    "bchusdt",
    "ltcusdt",
    "hbarusdt",
    "dotusdt",
    "nearusdt",
    "uniusdt",
    "aaveusdt",
    "wldusdt",
    "zecusdt",
]

# Build Trade and Depth Streams lists
trade_streams = [f"{symbol.lower()}@aggTrade" for symbol in SYMBOLS]
depth_streams = [f"{symbol.lower()}@depth@100ms" for symbol in SYMBOLS]

# Default WebSocket URLs using routed 2026 Binance segmentation paths
default_trade_url = "wss://fstream.binance.com/market/stream?streams=" + "/".join(trade_streams)
default_depth_url = "wss://fstream.binance.com/public/stream?streams=" + "/".join(depth_streams)

# WebSocket URLs with environment overrides
WS_TRADE_URL = os.getenv("WS_TRADE_URL", default_trade_url)
WS_DEPTH_URL = os.getenv("WS_DEPTH_URL", default_depth_url)

# Aggregation Configuration (in seconds)
WINDOWS = {
    "1m": 60,
    "5m": 300,
    "15m": 900
}

# Trade Metrics & Alerting Thresholds (V2.3 Normalized to USDT Notional)
LARGE_TRADE_NOTIONAL_USDT = 100000.0  # $100,000 notional size
TRADE_BURST_WINDOW = 10       # sliding window in seconds to check for trade bursts
TRADE_BURST_MULTIPLIER = 3.0   # trigger burst when trade count is > 3x recent average

# Order Book Imbalance Threshold
IMBALANCE_THRESHOLD = 0.15    # 15% bid/ask size difference

# Cooldown to avoid duplicate event spamming
EVENT_COOLDOWN_SECONDS = 60

# File Storage Paths
CSV_PATH = os.path.join(BASE_DIR, "flow_events.csv")
SWEEPS_CSV_PATH = "C:/Users/karan/Desktop/liquidity sweep/live_scanner/sweeps.csv"

# Dashboard Configurations
DASHBOARD_REFRESH_SECONDS = 1.0
WEB_DASHBOARD_PORT = int(os.getenv("WEB_DASHBOARD_PORT", "58536"))
MAX_RECENT_EVENTS = 100
MAX_DISPLAY_SYMBOLS = 20

# Scanner Constraints
EXECUTION_DISABLED = True

# AI Interpretation Configurations (V2.5)
AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
AI_INTERVAL_SECONDS = int(os.getenv("AI_INTERVAL_SECONDS", "120"))
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "20"))
AI_MAX_SYMBOLS = int(os.getenv("AI_MAX_SYMBOLS", "20"))
AI_CACHE_TTL_SECONDS = int(os.getenv("AI_CACHE_TTL_SECONDS", "60"))
AI_LOG_PROMPTS = os.getenv("AI_LOG_PROMPTS", "false").lower() == "true"
AI_SSL_VERIFY = os.getenv("AI_SSL_VERIFY", "true").lower() not in ("0", "false", "no", "off")
BINANCE_SSL_VERIFY = os.getenv("BINANCE_SSL_VERIFY", "true").lower() not in ("0", "false", "no", "off")

# API Keys and Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

# Event Recorder Configuration (v2.9)
RECORDING_ENABLED = os.getenv("RECORDING_ENABLED", "false").lower() == "true"

# TICS Stream Health & Data Integrity Guardian thresholds
TRADE_HEALTHY_MAX_SILENCE_MS = 1000.0
TRADE_STALE_AFTER_MS = 2500.0

DEPTH_HEALTHY_MAX_SILENCE_MS = 1000.0
DEPTH_STALE_AFTER_MS = 2500.0

LOCAL_BOOK_MAX_BUFFER = 5000

