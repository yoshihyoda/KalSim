"""Tests for SocialEnvironment module."""

import pytest
from datetime import datetime, timedelta

from src.core.social_environment import SocialEnvironment, MarketState


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_market_state_creation(self):
        """Test creating a MarketState with all fields."""
        state = MarketState(
            price=100.0,
            volume=1000000,
            short_interest=140.0,
            liquidity=1.0,
            trend="stable",
            timestamp=datetime(2021, 1, 11, 9, 0),
        )
        assert state.price == 100.0
        assert state.volume == 1000000
        assert state.short_interest == 140.0
        assert state.liquidity == 1.0
        assert state.trend == "stable"

    def test_market_state_price_change_calculation(self):
        """Test calculating price change percentage."""
        state = MarketState(
            price=120.0,
            volume=1000000,
            short_interest=140.0,
            liquidity=1.0,
            trend="rising",
            timestamp=datetime(2021, 1, 11, 10, 0),
            previous_price=100.0,
        )
        assert state.price_change_pct == pytest.approx(20.0)


class TestSocialEnvironment:
    """Tests for SocialEnvironment class."""

    def test_environment_initialization(self):
        """Test initializing SocialEnvironment with default state."""
        env = SocialEnvironment()
        assert env.current_state is not None
        assert env.current_state.price == 20.0

    def test_environment_custom_initialization(self, base_market_state):
        """Test initializing with custom market state."""
        env = SocialEnvironment(initial_state=base_market_state)
        assert env.current_state.price == 20.0
        assert env.current_state.volume == 10000000

    def test_update_market_state(self):
        """Test updating market state."""
        env = SocialEnvironment()
        initial_price = env.current_state.price

        env.update_state(price=50.0, trend="rising")

        assert env.current_state.price == 50.0
        assert env.current_state.trend == "rising"
        assert env.current_state.previous_price == initial_price

    def test_price_history_tracking(self):
        """Test that price history is tracked."""
        env = SocialEnvironment()
        env.update_state(price=25.0)
        env.update_state(price=30.0)
        env.update_state(price=35.0)

        assert len(env.price_history) == 4
        assert env.price_history[-1] == 35.0

    def test_get_price_trend(self):
        """Test getting price trend from history."""
        env = SocialEnvironment()
        env.update_state(price=21.0)
        env.update_state(price=22.0)
        env.update_state(price=23.0)

        trend = env.get_price_trend(window=3)
        assert trend == "rising"

    def test_liquidity_impact(self):
        """Test that high volume affects liquidity."""
        env = SocialEnvironment()
        env.update_state(volume=100000000)

        assert env.current_state.liquidity < 1.0

    def test_temporal_update(self):
        """Test advancing time in the environment."""
        env = SocialEnvironment()
        initial_time = env.current_state.timestamp

        env.advance_time(hours=1)

        assert env.current_state.timestamp == initial_time + timedelta(hours=1)

    def test_get_market_info_dict(self):
        """Test getting market state as dictionary."""
        env = SocialEnvironment()
        info = env.get_market_info()

        assert "price" in info
        assert "volume" in info
        assert "trend" in info
        assert "timestamp" in info
