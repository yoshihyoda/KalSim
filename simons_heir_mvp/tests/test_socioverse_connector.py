"""Tests for SocioVerseConnector module."""

import pytest
from unittest.mock import MagicMock, patch

from src.socioverse_connector import SocioVerseConnector


class TestSocioVerseConnector:
    """Tests for SocioVerseConnector class."""

    def test_init_without_token(self):
        """Test initialization without HF token."""
        connector = SocioVerseConnector()
        assert connector.hf_token is None or isinstance(connector.hf_token, str)

    def test_init_with_token(self):
        """Test initialization with HF token."""
        connector = SocioVerseConnector(hf_token="test_token")
        assert connector.hf_token == "test_token"

    def test_init_from_env(self, monkeypatch):
        """Test initialization reads HF_TOKEN from environment."""
        monkeypatch.setenv("HF_TOKEN", "env_token")
        connector = SocioVerseConnector()
        assert connector.hf_token == "env_token"

    def test_transform_user_basic(self):
        """Test user transformation with minimal data."""
        connector = SocioVerseConnector()
        
        user = {
            "user_id": "TestUser123",
            "AGE": "25-34",
            "GENDER": "Male",
            "Education": "Graduate",
            "Level of Consumption": "High",
            "influence": 0.75,
        }
        
        result = connector._transform_user(0, user)
        
        assert result["id"] == 0
        assert result["name"] == "TestUser123"
        assert "risk-seeking" in result["personality_traits"]
        assert result["beliefs"]["risk_tolerance"] == "high"
        assert result["beliefs"]["market_outlook"] == "bullish"
        assert result["social"]["influence_score"] == 0.75

    def test_transform_user_default_values(self):
        """Test user transformation with missing data."""
        connector = SocioVerseConnector()
        
        user = {}
        
        result = connector._transform_user(5, user)
        
        assert result["id"] == 5
        assert result["name"] == "SV_User_5"
        assert len(result["personality_traits"]) >= 2
        assert result["beliefs"]["risk_tolerance"] == "moderate"
        assert result["beliefs"]["market_outlook"] == "neutral"

    def test_extract_traits_young_user(self):
        """Test trait extraction for young users."""
        connector = SocioVerseConnector()
        
        user = {"AGE": "18-24", "Level of Consumption": "High"}
        traits = connector._extract_traits(user)
        
        assert "young" in traits
        assert "risk-seeking" in traits

    def test_extract_traits_experienced_user(self):
        """Test trait extraction for experienced users."""
        connector = SocioVerseConnector()
        
        user = {"AGE": "45-54", "Level of Consumption": "Low"}
        traits = connector._extract_traits(user)
        
        assert "experienced" in traits
        assert "cautious" in traits

    def test_extract_traits_analytical_user(self):
        """Test trait extraction for highly educated users."""
        connector = SocioVerseConnector()
        
        user = {"Education": "Master's Degree"}
        traits = connector._extract_traits(user)
        
        assert "analytical" in traits

    def test_extract_interests_young_user(self):
        """Test interest extraction for young users."""
        connector = SocioVerseConnector()
        
        user = {"AGE": "18-24"}
        interests = connector._extract_interests(user)
        
        assert "prediction markets" in interests
        assert "social media" in interests or "technology" in interests

    def test_extract_interests_older_user(self):
        """Test interest extraction for older users."""
        connector = SocioVerseConnector()
        
        user = {"AGE": "55-64"}
        interests = connector._extract_interests(user)
        
        assert "prediction markets" in interests
        assert "investing" in interests or "politics" in interests

    def test_map_consumption_to_risk_high(self):
        """Test consumption to risk mapping - high."""
        connector = SocioVerseConnector()
        
        assert connector._map_consumption_to_risk({"Level of Consumption": "High"}) == "high"

    def test_map_consumption_to_risk_low(self):
        """Test consumption to risk mapping - low."""
        connector = SocioVerseConnector()
        
        assert connector._map_consumption_to_risk({"Level of Consumption": "Low"}) == "low"

    def test_map_consumption_to_risk_moderate(self):
        """Test consumption to risk mapping - moderate."""
        connector = SocioVerseConnector()
        
        assert connector._map_consumption_to_risk({"Level of Consumption": "Medium"}) == "moderate"
        assert connector._map_consumption_to_risk({}) == "moderate"

    def test_infer_market_outlook(self):
        """Test market outlook inference."""
        connector = SocioVerseConnector()
        
        assert connector._infer_market_outlook({"Level of Consumption": "High"}) == "bullish"
        assert connector._infer_market_outlook({"Level of Consumption": "Low"}) == "bearish"
        assert connector._infer_market_outlook({}) == "neutral"

    def test_infer_trust_level(self):
        """Test institutional trust inference."""
        connector = SocioVerseConnector()
        
        assert connector._infer_trust_level({"Education": "Master's Degree"}) == "moderate"
        assert connector._infer_trust_level({"Education": "High School"}) == "low"
        assert connector._infer_trust_level({}) == "moderate"

    def test_fetch_user_pool_success(self):
        """Test successful fetch from SocioVerse dataset."""
        mock_dataset = [
            {"user_id": "User1", "AGE": "25-34", "influence": 0.5},
            {"user_id": "User2", "AGE": "35-44", "influence": 0.7},
        ]
        
        mock_datasets = MagicMock()
        mock_datasets.load_dataset = MagicMock(return_value=mock_dataset)
        
        connector = SocioVerseConnector(hf_token="test_token")
        
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            personas = connector.fetch_user_pool(count=2)
        
        assert len(personas) == 2
        assert personas[0]["name"] == "User1"
        assert personas[1]["name"] == "User2"

    def test_fetch_user_pool_failure(self):
        """Test graceful failure when fetch fails."""
        mock_datasets = MagicMock()
        mock_datasets.load_dataset = MagicMock(side_effect=Exception("Access denied"))
        
        connector = SocioVerseConnector()
        
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            personas = connector.fetch_user_pool(count=10)
        
        assert personas == []

    def test_fetch_user_pool_no_datasets_library(self):
        """Test handling when datasets library not installed."""
        connector = SocioVerseConnector()
        
        with patch.dict("sys.modules", {"datasets": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                personas = connector.fetch_user_pool(count=5)
                assert personas == []
