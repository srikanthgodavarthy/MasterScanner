import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get("SCAN_CACHE_TTL", 300))  # 5 min
    # Toggle: use the real pandas scan engine (needs network + yfinance) or
    # deterministic sample data (for local UI dev / offline demo).
    USE_LIVE_SCAN = os.environ.get("USE_LIVE_SCAN", "0") == "1"
