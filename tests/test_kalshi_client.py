"""Unit tests for KalshiClient behavior."""

import time
from unittest.mock import MagicMock, patch

from src.kalshi import KalshiClient


class TestKalshiClientCaching:
    """Tests for KalshiClient caching behavior."""

    def test_successive_calls_use_cache(self):
        """Test that rapid successive calls return cached data."""
        client = KalshiClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "markets": [{"ticker": "TEST", "volume_24h": 100}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response) as mock_get:
            result1 = client.get_public_markets(limit=5, check_exchange_status=False)
            result2 = client.get_public_markets(limit=5, check_exchange_status=False)

            assert mock_get.call_count == 1
            assert result1 == result2

    def test_different_params_bypass_cache(self):
        """Test that different parameters create separate cache entries."""
        client = KalshiClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "markets": [{"ticker": "TEST", "volume_24h": 100}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response) as mock_get:
            client.get_public_markets(limit=5, check_exchange_status=False)
            client.get_public_markets(limit=10, check_exchange_status=False)

            assert mock_get.call_count == 2

    def test_cache_expires_after_ttl(self):
        """Test that cache expires after the TTL period."""
        client = KalshiClient()
        client._cache_expiry_seconds = 0.1

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "markets": [{"ticker": "TEST", "volume_24h": 100}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response) as mock_get:
            client.get_public_markets(limit=5, check_exchange_status=False)
            time.sleep(0.15)
            client.get_public_markets(limit=5, check_exchange_status=False)

            assert mock_get.call_count == 2

    def test_cache_returns_same_data(self):
        """Test that cached data is identical to original."""
        client = KalshiClient()

        expected_markets = [
            {"ticker": "MKT1", "volume_24h": 500},
            {"ticker": "MKT2", "volume_24h": 300},
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"markets": expected_markets}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response):
            result1 = client.get_public_markets(limit=5, check_exchange_status=False)
            result2 = client.get_public_markets(limit=5, check_exchange_status=False)

            assert result1 == result2
            assert result1 == expected_markets[:5]
