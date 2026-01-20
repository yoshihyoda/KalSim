"""Tests for SocialEnvironment Kalshi integration."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.social_environment import SocialEnvironment


class TestSocialEnvironmentKalshi:
    """Tests for Kalshi integration in SocialEnvironment."""

    def test_init_without_kalshi(self):
        """Test initialization without Kalshi."""
        env = SocialEnvironment(use_kalshi=False)
        assert env._kalshi_client is None
        assert env._kalshi_trends is None
        assert env._use_kalshi is False

    def test_init_with_kalshi(self):
        """Test initialization with Kalshi enabled."""
        env = SocialEnvironment(use_kalshi=True)
        assert env._kalshi_client is not None
        assert env._use_kalshi is True

    def test_load_kalshi_trends_without_client(self):
        """Test load_kalshi_trends returns empty when no client."""
        env = SocialEnvironment(use_kalshi=False)
        result = env.load_kalshi_trends()
        assert result == {}

    def test_load_kalshi_trends_with_mock(self):
        """Test load_kalshi_trends with mocked client."""
        env = SocialEnvironment(use_kalshi=True)
        
        mock_events = [
            {"title": "Will Bitcoin hit $100k?", "event_ticker": "BTC100K"},
            {"title": "2024 Election Winner", "event_ticker": "PRES2024"},
        ]
        mock_analysis = {
            "topics": ["Bitcoin $100k", "2024 Election"],
            "tickers": ["BTC100K", "PRES2024"],
            "summary": "Top trending markets",
        }
        
        env._kalshi_client.get_trending_events = MagicMock(return_value=mock_events)
        env._kalshi_client.analyze_trends = MagicMock(return_value=mock_analysis)
        
        result = env.load_kalshi_trends(limit=5)
        
        assert result == mock_analysis
        assert env._kalshi_trends == mock_analysis
        env._kalshi_client.get_trending_events.assert_called_once_with(limit=5)

    def test_get_trending_topics_empty(self):
        """Test get_trending_topics returns empty list when no data."""
        env = SocialEnvironment(use_kalshi=False)
        assert env.get_trending_topics() == []

    def test_get_trending_topics_with_data(self):
        """Test get_trending_topics returns topics from loaded data."""
        env = SocialEnvironment(use_kalshi=False)
        env._kalshi_trends = {
            "topics": ["Topic A", "Topic B", "Topic C"],
            "summary": "Test summary",
        }
        
        topics = env.get_trending_topics()
        assert topics == ["Topic A", "Topic B", "Topic C"]

    def test_get_kalshi_summary_empty(self):
        """Test get_kalshi_summary returns empty when no data."""
        env = SocialEnvironment(use_kalshi=False)
        assert env.get_kalshi_summary() == ""

    def test_get_kalshi_summary_with_data(self):
        """Test get_kalshi_summary returns summary from loaded data."""
        env = SocialEnvironment(use_kalshi=False)
        env._kalshi_trends = {
            "topics": ["Topic A"],
            "summary": "Market summary text",
        }
        
        assert env.get_kalshi_summary() == "Market summary text"

    def test_has_kalshi_data_false_when_empty(self):
        """Test has_kalshi_data returns False when no data."""
        env = SocialEnvironment(use_kalshi=False)
        assert env.has_kalshi_data is False

    def test_has_kalshi_data_false_when_no_topics(self):
        """Test has_kalshi_data returns False when topics empty."""
        env = SocialEnvironment(use_kalshi=False)
        env._kalshi_trends = {"topics": [], "summary": ""}
        assert env.has_kalshi_data is False

    def test_has_kalshi_data_true_when_loaded(self):
        """Test has_kalshi_data returns True when data loaded."""
        env = SocialEnvironment(use_kalshi=False)
        env._kalshi_trends = {"topics": ["Topic A"], "summary": "Summary"}
        assert env.has_kalshi_data is True

    def test_kalshi_integration_with_existing_state(self, base_market_state):
        """Test Kalshi integration works with existing market state."""
        env = SocialEnvironment(initial_state=base_market_state, use_kalshi=True)
        
        assert env.current_state.price == 20.0
        assert env._kalshi_client is not None

    def test_load_kalshi_trends_handles_exception(self):
        """Test load_kalshi_trends handles API errors gracefully."""
        env = SocialEnvironment(use_kalshi=True)
        
        env._kalshi_client.get_trending_events = MagicMock(
            side_effect=Exception("API Error")
        )
        
        result = env.load_kalshi_trends()
        
        assert result == {}
        assert env._kalshi_trends is None
