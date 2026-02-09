"""Tests for Simulation live data integration."""

import pytest
from unittest.mock import MagicMock, patch

from src.simulation import Simulation
from src.interfaces import MarketDataProviderABC


class TestSimulationPersonaLoading:
    """Tests for persona loading priority chain."""

    def test_priority_1_custom_agents(self):
        """Test that custom_agents has highest priority."""
        custom = [
            {"id": 0, "name": "CustomAgent1"},
            {"id": 1, "name": "CustomAgent2"},
        ]
        
        sim = Simulation(
            days=1,
            agent_count=2,
            mock_llm=True,
            use_kalshi=True,
            custom_agents=custom,
        )
        
        personas = sim._load_personas()
        
        assert len(personas) == 2
        assert personas[0]["name"] == "CustomAgent1"

    @patch("src.simulation.Simulation._try_load_socioverse")
    def test_priority_2_socioverse(self, mock_socioverse):
        """Test that SocioVerse is tried when use_kalshi is True."""
        mock_personas = [{"id": 0, "name": "SV_User_0"}]
        mock_socioverse.return_value = mock_personas
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
        )
        
        personas = sim._load_personas()
        
        mock_socioverse.assert_called_once()
        assert personas[0]["name"] == "SV_User_0"

    @patch("src.simulation.Simulation._try_load_socioverse")
    @patch("src.simulation.Simulation._try_generate_from_kalshi")
    def test_priority_3_kalshi_generation(self, mock_kalshi_gen, mock_socioverse):
        """Test Kalshi generation when SocioVerse fails."""
        mock_socioverse.return_value = None
        mock_kalshi_gen.return_value = [{"id": 0, "name": "KalshiAgent"}]
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
        )
        sim._kalshi_analysis = {"summary": "Test trends"}
        
        personas = sim._load_personas()
        
        mock_kalshi_gen.assert_called_once()
        assert personas[0]["name"] == "KalshiAgent"

    @patch("src.simulation.Simulation._try_load_socioverse")
    def test_priority_4_default_generation(self, mock_socioverse):
        """Test default persona generation when live sources fail.
        
        Note: Static file loading has been deprecated. The simulation now
        falls back to generated defaults when live sources are unavailable.
        """
        mock_socioverse.return_value = None
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
        )
        sim._kalshi_analysis = None
        
        personas = sim._load_personas()
        
        assert len(personas) == 1
        assert "name" in personas[0]
        assert personas[0]["name"].startswith("User_")

    @patch("src.simulation.Simulation._try_load_socioverse")
    def test_default_generation_respects_agent_count(self, mock_socioverse):
        """Test default persona generation as last resort."""
        mock_socioverse.return_value = None
        
        sim = Simulation(
            days=1,
            agent_count=3,
            mock_llm=True,
            use_kalshi=True,
        )
        sim._kalshi_analysis = None
        sim.persona_file = MagicMock()
        sim.persona_file.exists.return_value = False
        
        personas = sim._load_personas()
        
        assert len(personas) == 3
        assert all("name" in p for p in personas)

    def test_socioverse_not_tried_when_use_kalshi_false(self):
        """Test SocioVerse is skipped when use_kalshi is False."""
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=False,
        )
        sim.persona_file = MagicMock()
        sim.persona_file.exists.return_value = False
        
        with patch.object(sim, "_try_load_socioverse") as mock_sv:
            sim._load_personas()
            mock_sv.assert_not_called()


class TestSimulationSeedTweets:
    """Tests for dynamic seed tweet generation."""

    def test_dynamic_content_from_kalshi(self):
        """Test dynamic content is generated from Kalshi topics."""
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
        )
        sim._kalshi_analysis = {
            "topics": ["Bitcoin Price", "Election Odds"],
            "summary": "Test",
        }
        sim.tweets_file = MagicMock()
        sim.tweets_file.exists.return_value = False
        
        tweets = sim._load_seed_tweets()
        
        assert len(tweets) > 0
        assert any("Bitcoin Price" in t for t in tweets)
        assert any("Election Odds" in t for t in tweets)

    def test_kalshi_content_uses_templates(self):
        """Test Kalshi content uses various templates."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        
        topics = ["Test Topic"]
        content = sim._generate_kalshi_based_content(topics)
        
        assert len(content) == 10
        assert any("What do you think" in c for c in content)
        assert any("bullish" in c for c in content)

    def test_fallback_to_generated_when_no_kalshi(self):
        """Test falls back to generated content when Kalshi unavailable.
        
        Note: Static file loading has been deprecated. The simulation now
        generates sample content when live sources are unavailable.
        """
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=False,
        )
        
        tweets = sim._load_seed_tweets()
        
        assert len(tweets) > 0
        assert any("GME" in t for t in tweets)

    def test_fallback_to_generated_when_no_file(self):
        """Test falls back to generated when no file."""
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=False,
        )
        sim.tweets_file = MagicMock()
        sim.tweets_file.exists.return_value = False
        
        tweets = sim._load_seed_tweets()
        
        assert len(tweets) > 0
        assert any("GME" in t for t in tweets)


class TestTryLoadSocioverse:
    """Tests for _try_load_socioverse method."""

    def test_returns_personas_on_success(self):
        """Test returns personas when SocioVerse fetch succeeds."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        
        with patch("src.socioverse_connector.SocioVerseConnector") as mock_class:
            mock_connector = MagicMock()
            mock_connector.fetch_user_pool.return_value = [{"name": "SV_User"}]
            mock_class.return_value = mock_connector
            
            result = sim._try_load_socioverse()
        
        assert result == [{"name": "SV_User"}]

    def test_returns_none_on_empty(self):
        """Test returns None when SocioVerse returns empty."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        
        with patch("src.socioverse_connector.SocioVerseConnector") as mock_class:
            mock_connector = MagicMock()
            mock_connector.fetch_user_pool.return_value = []
            mock_class.return_value = mock_connector
            
            result = sim._try_load_socioverse()
        
        assert result is None

    def test_returns_none_on_exception(self):
        """Test returns None when exception occurs."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        
        with patch("src.socioverse_connector.SocioVerseConnector") as mock_class:
            mock_class.side_effect = Exception("Connection failed")
            
            result = sim._try_load_socioverse()
        
        assert result is None


class TestUpdateMarketStateLive:
    """Tests for live market state updates via Kalshi API."""

    def test_update_market_state_calls_kalshi_when_live(self):
        """Test that _update_market_state calls Kalshi API when use_kalshi is True."""
        mock_client = MagicMock(spec=MarketDataProviderABC)
        mock_client.get_public_markets.return_value = [
            {"ticker": "TEST-MKT", "yes_price": 0.55, "volume_24h": 12345}
        ]
        mock_client.get_trending_events.return_value = []
        mock_client.analyze_trends.return_value = {"topics": [], "summary": ""}
        
        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_client,
            market_topic="TEST-MKT",
        )
        sim.setup()
        
        sim._update_market_state(step=1)
        
        mock_client.get_public_markets.assert_called()
        assert sim._current_price == pytest.approx(55.0)

    def test_update_market_state_updates_price_history(self):
        """Test that live updates append to price history."""
        mock_client = MagicMock(spec=MarketDataProviderABC)
        mock_client.get_public_markets.return_value = [
            {"ticker": "TEST-MKT", "yes_price": 0.65, "volume_24h": 5000}
        ]
        mock_client.get_trending_events.return_value = []
        mock_client.analyze_trends.return_value = {"topics": [], "summary": ""}
        
        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_client,
            market_topic="TEST-MKT",
        )
        sim.setup()
        initial_history_len = len(sim._price_history)
        
        sim._update_market_state(step=1)
        
        assert len(sim._price_history) == initial_history_len + 1
        assert sim._price_history[-1] == pytest.approx(65.0)

    def test_update_market_state_falls_back_when_market_not_found(self):
        """Test fallback to formula when target market not in response."""
        mock_client = MagicMock(spec=MarketDataProviderABC)
        mock_client.get_public_markets.return_value = [
            {"ticker": "OTHER-MKT", "yes_price": 0.80, "volume_24h": 1000}
        ]
        mock_client.get_trending_events.return_value = []
        mock_client.analyze_trends.return_value = {"topics": [], "summary": ""}
        
        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_client,
            market_topic="TEST-MKT",
        )
        sim.setup()
        initial_price = sim._current_price
        
        sim._update_market_state(step=1)
        
        assert sim._current_price != 80.0
        assert sim._current_price != initial_price

    def test_update_market_state_matches_market_title(self):
        """Test human-readable topic can match market title from list."""
        mock_client = MagicMock()
        mock_client.get_public_markets.return_value = [
            {
                "ticker": "TV-PRISONBREAK",
                "title": "When will Prison Break return?",
                "yes_price": 0.42,
                "volume_24h": 1000,
            }
        ]
        mock_client.get_trending_events.return_value = []
        mock_client.analyze_trends.return_value = {"topics": [], "summary": ""}
        mock_client.get_market.return_value = None

        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_client,
            market_topic="When will Prison Break return?",
        )
        sim.setup()

        sim._update_market_state(step=1)

        assert sim._current_price == pytest.approx(42.0)
        mock_client.get_market.assert_not_called()

    def test_update_market_state_skips_direct_fetch_for_non_ticker_topic(self):
        """Test non-ticker topics do not trigger direct get_market API calls."""
        mock_client = MagicMock()
        mock_client.get_public_markets.return_value = [
            {"ticker": "OTHER-MKT", "yes_price": 0.80, "volume_24h": 1000}
        ]
        mock_client.get_trending_events.return_value = []
        mock_client.analyze_trends.return_value = {"topics": [], "summary": ""}
        mock_client.get_market.return_value = None

        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_client,
            market_topic="When will Prison Break return?",
        )
        sim.setup()

        sim._update_market_state(step=1)

        mock_client.get_market.assert_not_called()

    def test_update_market_state_falls_back_on_empty_response(self):
        """Test fallback to formula when API returns empty list."""
        mock_client = MagicMock(spec=MarketDataProviderABC)
        mock_client.get_public_markets.return_value = []
        mock_client.get_trending_events.return_value = []
        mock_client.analyze_trends.return_value = {"topics": [], "summary": ""}
        
        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_client,
            market_topic="TEST-MKT",
        )
        sim.setup()
        initial_price = sim._current_price
        
        sim._update_market_state(step=1)
        
        assert sim._current_price != initial_price

    def test_update_market_state_uses_formula_when_kalshi_disabled(self):
        """Test that formula is used when use_kalshi is False."""
        sim = Simulation(
            agent_count=1,
            days=1,
            mock_llm=True,
            use_kalshi=False,
        )
        sim.setup()
        initial_price = sim._current_price
        
        sim._update_market_state(step=1)
        
        assert sim._current_price != initial_price
        assert sim._kalshi_client is None


class TestTryGenerateFromKalshi:
    """Tests for _try_generate_from_kalshi method."""

    def test_generates_personas_from_trends(self):
        """Test generates personas from Kalshi trends."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        sim._kalshi_analysis = {"summary": "Bitcoin trends"}
        sim.llm = MagicMock()
        
        with patch("src.agent_generator.AgentGenerator") as mock_class:
            mock_generator = MagicMock()
            mock_generator.generate_agents.return_value = [{"name": "GeneratedAgent"}]
            mock_class.return_value = mock_generator
            
            result = sim._try_generate_from_kalshi()
        
        assert result == [{"name": "GeneratedAgent"}]

    def test_returns_none_when_no_summary(self):
        """Test returns None when no trend summary."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        sim._kalshi_analysis = {"topics": ["test"]}
        
        result = sim._try_generate_from_kalshi()
        
        assert result is None

    def test_returns_none_when_no_kalshi_analysis(self):
        """Test returns None when kalshi_analysis is None."""
        sim = Simulation(days=1, agent_count=1, mock_llm=True)
        sim._kalshi_analysis = None
        
        result = sim._try_generate_from_kalshi()
        
        assert result is None
