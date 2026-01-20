"""Tests for Simulation Kalshi integration."""

import pytest
from unittest.mock import MagicMock, patch

from src.simulation import Simulation


class TestSimulationKalshi:
    """Tests for Kalshi integration in Simulation."""

    def test_init_without_kalshi(self):
        """Test initialization without Kalshi."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=False)
        assert sim._kalshi_client is None
        assert sim.use_kalshi is False

    def test_init_with_kalshi(self):
        """Test initialization with Kalshi enabled."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=True)
        assert sim._kalshi_client is not None
        assert sim.use_kalshi is True

    def test_real_llm_enables_kalshi(self):
        """Test that disabling mock_llm enables Kalshi by default."""
        sim = Simulation(days=1, agent_count=3, mock_llm=False)
        assert sim.use_kalshi is True

    def test_mock_llm_no_auto_kalshi(self):
        """Test that mock_llm doesn't auto-enable Kalshi."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True)
        assert sim.use_kalshi is False

    @patch("src.simulation.KalshiClient")
    def test_setup_loads_kalshi_trends(self, mock_kalshi_class):
        """Test that setup loads Kalshi trends when enabled."""
        mock_client = MagicMock()
        mock_client.get_trending_events.return_value = [
            {"title": "Test Event", "event_ticker": "TEST"}
        ]
        mock_client.analyze_trends.return_value = {
            "topics": ["Test Topic"],
            "summary": "Test summary",
        }
        mock_kalshi_class.return_value = mock_client
        
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=True)
        sim._kalshi_client = mock_client
        sim.setup()
        
        mock_client.get_trending_events.assert_called_once()
        mock_client.analyze_trends.assert_called_once()

    def test_get_social_info_uses_kalshi_topics(self):
        """Test that _get_social_info uses Kalshi topics when available."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=True)
        sim._kalshi_analysis = {
            "topics": ["Kalshi Topic 1", "Kalshi Topic 2", "Kalshi Topic 3"],
            "summary": "Real market data",
        }
        
        social_info = sim._get_social_info()
        
        assert "Kalshi Topic 1" in social_info.trending_topics
        assert "Kalshi Topic 2" in social_info.trending_topics

    def test_get_social_info_falls_back_to_gme(self):
        """Test that _get_social_info falls back to GME when no Kalshi data."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=False)
        
        social_info = sim._get_social_info()
        
        assert "$GME" in social_info.trending_topics
        assert "GameStop" in social_info.trending_topics

    def test_get_social_info_empty_kalshi_falls_back(self):
        """Test fallback when Kalshi analysis is empty."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=True)
        sim._kalshi_analysis = {"topics": [], "summary": ""}
        
        social_info = sim._get_social_info()
        
        assert "$GME" in social_info.trending_topics

    @patch("src.simulation.KalshiClient")
    def test_load_kalshi_trends_handles_errors(self, mock_kalshi_class):
        """Test that _load_kalshi_trends handles API errors."""
        mock_client = MagicMock()
        mock_client.get_trending_events.side_effect = Exception("API Error")
        mock_kalshi_class.return_value = mock_client
        
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=True)
        sim._kalshi_client = mock_client
        
        sim._load_kalshi_trends()
        
        assert sim._kalshi_analysis is None

    def test_full_simulation_with_kalshi_mock(self):
        """Test running a full simulation with mocked Kalshi data."""
        sim = Simulation(days=1, agent_count=3, mock_llm=True, use_kalshi=True)
        
        sim._kalshi_analysis = {
            "topics": ["Bitcoin Price", "Election Odds", "Fed Rate"],
            "summary": "Test market summary",
        }
        
        results = sim.run()
        
        assert len(results) > 0
        assert sim.use_kalshi is True
