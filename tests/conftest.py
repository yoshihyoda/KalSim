"""Shared pytest fixtures for simons_heir_mvp tests."""

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface that returns predictable responses."""
    mock = MagicMock()
    mock.generate.return_value = "ACTION: HOLD\nCONTENT: Monitoring the situation."
    mock.health_check.return_value = True
    mock.mock_mode = True
    return mock


@pytest.fixture
def sample_persona() -> dict[str, Any]:
    """Sample persona for testing."""
    return {
        "id": 0,
        "name": "TestUser",
        "personality_traits": ["risk-seeking", "impulsive", "optimistic"],
        "interests": ["stocks", "crypto", "reddit"],
        "beliefs": {
            "risk_tolerance": "high",
            "market_outlook": "bullish",
            "trust_in_institutions": "low",
        },
        "social": {"follower_count": 1000, "influence_score": 0.7},
    }


@pytest.fixture
def sample_personas() -> list[dict[str, Any]]:
    """Multiple sample personas for testing."""
    return [
        {
            "id": 0,
            "name": "DiamondHandsDave",
            "personality_traits": ["risk-seeking", "impulsive", "optimistic"],
            "interests": ["stocks", "crypto", "reddit"],
            "beliefs": {
                "risk_tolerance": "high",
                "market_outlook": "bullish",
                "trust_in_institutions": "low",
            },
            "social": {"follower_count": 1523, "influence_score": 0.72},
        },
        {
            "id": 1,
            "name": "CautiousCarla",
            "personality_traits": ["cautious", "analytical", "skeptical"],
            "interests": ["finance", "economics", "data analysis"],
            "beliefs": {
                "risk_tolerance": "low",
                "market_outlook": "neutral",
                "trust_in_institutions": "moderate",
            },
            "social": {"follower_count": 892, "influence_score": 0.45},
        },
        {
            "id": 2,
            "name": "MoonBoi_Mike",
            "personality_traits": ["emotional", "reactive", "enthusiastic"],
            "interests": ["memes", "trading", "WSB"],
            "beliefs": {
                "risk_tolerance": "high",
                "market_outlook": "bullish",
                "trust_in_institutions": "low",
            },
            "social": {"follower_count": 5234, "influence_score": 0.85},
        },
    ]


@pytest.fixture
def base_market_state() -> dict[str, Any]:
    """Base market state for testing."""
    return {
        "price": 20.0,
        "volume": 10000000,
        "short_interest": 140.0,
        "liquidity": 1.0,
        "trend": "stable",
        "timestamp": datetime(2021, 1, 11, 9, 0),
    }


@pytest.fixture
def surge_market_state() -> dict[str, Any]:
    """Market state during a price surge."""
    return {
        "price": 350.0,
        "volume": 100000000,
        "short_interest": 140.0,
        "liquidity": 0.3,
        "trend": "surging",
        "timestamp": datetime(2021, 1, 27, 14, 0),
    }


@pytest.fixture
def sample_tweet() -> dict[str, Any]:
    """Sample tweet/post for testing."""
    return {
        "id": "tweet_001",
        "author_id": 0,
        "content": "$GME to the moon! Diamond hands forever!",
        "timestamp": datetime(2021, 1, 25, 10, 30),
        "upvotes": 1500,
        "sentiment": 0.9,
    }


@pytest.fixture
def sample_network_edges() -> list[tuple[int, int]]:
    """Sample social network edges for testing."""
    return [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
    ]
