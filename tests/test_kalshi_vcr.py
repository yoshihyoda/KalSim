"""VCR-recorded integration tests for Kalshi API.

These tests use recorded HTTP responses to test real API behavior
without making actual network requests during test runs.

To record new cassettes, run with:
    pytest tests/test_kalshi_vcr.py --vcr-record=new_episodes

Or set KALSHI_VCR_RECORD=1 environment variable.
"""

import os
import pytest
from pathlib import Path

from src.kalshi import KalshiClient

CASSETTES_DIR = Path(__file__).parent / "cassettes"
RECORD_MODE = "new_episodes" if os.environ.get("KALSHI_VCR_RECORD") else "none"


@pytest.fixture
def kalshi_client():
    """Create a Kalshi client for testing."""
    return KalshiClient()


def cassette_exists(name: str) -> bool:
    """Check if a cassette file exists."""
    return (CASSETTES_DIR / f"{name}.yaml").exists()


@pytest.mark.vcr(cassette_library_dir=str(CASSETTES_DIR), record_mode=RECORD_MODE)
class TestKalshiClientVCR:
    """VCR-recorded tests for KalshiClient.
    
    These tests require cassettes to be recorded first. If cassettes don't exist
    and KALSHI_VCR_RECORD is not set, tests will be skipped.
    """

    @pytest.mark.skipif(
        not cassette_exists("TestKalshiClientVCR.test_get_exchange_status") and RECORD_MODE == "none",
        reason="Cassette not recorded. Set KALSHI_VCR_RECORD=1 to record."
    )
    def test_get_exchange_status(self, kalshi_client):
        """Test fetching exchange status."""
        status = kalshi_client.get_exchange_status()
        
        if status is not None:
            assert isinstance(status, dict)
            assert "exchange_active" in status or "trading_active" in status

    @pytest.mark.skipif(
        not cassette_exists("TestKalshiClientVCR.test_get_trending_events") and RECORD_MODE == "none",
        reason="Cassette not recorded. Set KALSHI_VCR_RECORD=1 to record."
    )
    def test_get_trending_events(self, kalshi_client):
        """Test fetching trending events from Kalshi."""
        events = kalshi_client.get_trending_events(limit=5)
        
        assert isinstance(events, list)
        if events:
            event = events[0]
            assert "title" in event or "event_ticker" in event

    @pytest.mark.skipif(
        not cassette_exists("TestKalshiClientVCR.test_get_public_markets") and RECORD_MODE == "none",
        reason="Cassette not recorded. Set KALSHI_VCR_RECORD=1 to record."
    )
    def test_get_public_markets(self, kalshi_client):
        """Test fetching public markets."""
        markets = kalshi_client.get_public_markets(limit=5)
        
        assert isinstance(markets, list)
        if markets:
            market = markets[0]
            assert "ticker" in market or "title" in market

    @pytest.mark.skipif(
        not cassette_exists("TestKalshiClientVCR.test_analyze_trends") and RECORD_MODE == "none",
        reason="Cassette not recorded. Set KALSHI_VCR_RECORD=1 to record."
    )
    def test_analyze_trends(self, kalshi_client):
        """Test trend analysis from events."""
        events = kalshi_client.get_trending_events(limit=5)
        analysis = kalshi_client.analyze_trends(events)
        
        assert isinstance(analysis, dict)
        assert "topics" in analysis
        assert "summary" in analysis
        assert isinstance(analysis["topics"], list)

    @pytest.mark.skipif(
        not cassette_exists("TestKalshiClientVCR.test_summarize_event") and RECORD_MODE == "none",
        reason="Cassette not recorded. Set KALSHI_VCR_RECORD=1 to record."
    )
    def test_summarize_event(self, kalshi_client):
        """Test event summarization."""
        events = kalshi_client.get_trending_events(limit=1)
        
        if events:
            summary = kalshi_client.summarize_event(events[0])
            assert isinstance(summary, str)
            assert len(summary) > 0


class TestKalshiClientMocked:
    """Tests for KalshiClient with mocked responses (no VCR)."""

    def test_client_initialization(self):
        """Test client can be initialized."""
        client = KalshiClient()
        assert client is not None
        assert hasattr(client, "session")
        assert hasattr(client, "_market_cache")
        assert client._cache_expiry_seconds == 30

    def test_analyze_trends_empty_events(self, kalshi_client):
        """Test trend analysis with empty events list."""
        analysis = kalshi_client.analyze_trends([])
        
        assert isinstance(analysis, dict)
        assert "topics" in analysis
        assert analysis["topics"] == ["General Market"]

    def test_analyze_trends_with_mock_events(self, kalshi_client):
        """Test trend analysis with mock event data."""
        mock_events = [
            {
                "title": "Will Bitcoin hit $100k?",
                "event_ticker": "BITCOIN100K",
                "series_ticker": "CRYPTO",
                "markets": [
                    {"title": "Yes", "volume_24h": 10000}
                ],
            },
            {
                "title": "Election 2024 Winner",
                "event_ticker": "ELECTION2024",
                "series_ticker": "POLITICS",
                "markets": [
                    {"title": "Candidate A", "volume_24h": 50000}
                ],
            },
        ]
        
        analysis = kalshi_client.analyze_trends(mock_events)
        
        assert len(analysis["topics"]) >= 2
        assert "summary" in analysis
