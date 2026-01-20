"""Configuration module for Simons' Heir MVP.

Manages project-wide constants, paths, and simulation parameters.
"""

from pathlib import Path
from typing import Final

# Project paths
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.resolve()
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"

# Data files (DEPRECATED: Use live data sources instead)
# These paths are kept for backward compatibility but are no longer used by default.
# The simulation now prioritizes SocioVerse/Kalshi live data.
PERSONA_FILE: Final[Path] = DATA_DIR / "user_pool_100.json"  # Deprecated
TWEETS_FILE: Final[Path] = DATA_DIR / "gamestop_tweets.csv"  # Deprecated

# Result files
SIMULATION_LOG_FILE: Final[Path] = RESULTS_DIR / "simulation_log.json"
SENTIMENT_PLOT_FILE: Final[Path] = RESULTS_DIR / "sentiment_trend.png"

# Simulation parameters
SIMULATION_DAYS: Final[int] = 7
AGENTS_COUNT: Final[int] = 100
TIME_STEP_HOURS: Final[int] = 1
STEPS_PER_DAY: Final[int] = 24 // TIME_STEP_HOURS

# Ollama LLM configuration
OLLAMA_HOST: Final[str] = "localhost"
OLLAMA_PORT: Final[int] = 11434
OLLAMA_MODEL: Final[str] = "llama3.1:8b"
OLLAMA_BASE_URL: Final[str] = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_GENERATE_ENDPOINT: Final[str] = f"{OLLAMA_BASE_URL}/api/generate"

# LLM request settings
LLM_TIMEOUT_SECONDS: Final[int] = 120
LLM_MAX_RETRIES: Final[int] = 3
LLM_DEFAULT_TEMPERATURE: Final[float] = 0.7

# Sentiment analysis keywords
POSITIVE_KEYWORDS: Final[list[str]] = [
    "moon", "rocket", "diamond hands", "hold", "buy", "bullish",
    "to the moon", "🚀", "💎", "gains", "squeeze", "yolo",
    "tendies", "apes", "strong", "winning", "up"
]

NEGATIVE_KEYWORDS: Final[list[str]] = [
    "sell", "crash", "dump", "bearish", "loss", "paper hands",
    "falling", "down", "fear", "panic", "drop", "red",
    "bleeding", "rip", "dead"
]

TRACKED_KEYWORDS: Final[list[str]] = [
    "$GME", "GME", "GameStop", "to the moon", "diamond hands",
    "paper hands", "squeeze", "short squeeze", "hold the line",
    "apes together", "YOLO", "tendies", "Robinhood", "hedge fund"
]


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
