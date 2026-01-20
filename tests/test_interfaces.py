"""Tests for ABC interfaces and dependency injection."""

import pytest
from unittest.mock import MagicMock

from src.interfaces import LLMInterfaceABC, MarketDataProviderABC, UserPoolProviderABC
from src.llm_interface import LlamaInterface
from src.kalshi import KalshiClient
from src.socioverse_connector import SocioVerseConnector
from src.simulation import Simulation


class TestInterfaceImplementations:
    """Test that concrete classes implement ABCs correctly."""

    def test_llama_interface_implements_abc(self):
        """Test LlamaInterface implements LLMInterfaceABC."""
        assert issubclass(LlamaInterface, LLMInterfaceABC)
        
        llm = LlamaInterface(mock_mode=True)
        assert isinstance(llm, LLMInterfaceABC)
        
        assert hasattr(llm, "generate")
        assert hasattr(llm, "health_check")

    def test_kalshi_client_implements_abc(self):
        """Test KalshiClient implements MarketDataProviderABC."""
        assert issubclass(KalshiClient, MarketDataProviderABC)
        
        client = KalshiClient()
        assert isinstance(client, MarketDataProviderABC)
        
        assert hasattr(client, "get_trending_events")
        assert hasattr(client, "analyze_trends")

    def test_socioverse_connector_implements_abc(self):
        """Test SocioVerseConnector implements UserPoolProviderABC."""
        assert issubclass(SocioVerseConnector, UserPoolProviderABC)
        
        connector = SocioVerseConnector()
        assert isinstance(connector, UserPoolProviderABC)
        
        assert hasattr(connector, "fetch_user_pool")


class TestSimulationDependencyInjection:
    """Test dependency injection in Simulation class."""

    def test_inject_llm_provider(self):
        """Test injecting custom LLM provider."""
        mock_llm = MagicMock(spec=LLMInterfaceABC)
        mock_llm.generate.return_value = "ACTION: HOLD\nCONTENT: Test"
        mock_llm.health_check.return_value = True
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            llm_provider=mock_llm,
        )
        
        assert sim.llm is mock_llm
        
        sim.setup()
        assert sim.llm is mock_llm

    def test_inject_market_provider(self):
        """Test injecting custom market data provider."""
        mock_market = MagicMock(spec=MarketDataProviderABC)
        mock_market.get_trending_events.return_value = [
            {"title": "Test Event", "event_ticker": "TEST"}
        ]
        mock_market.analyze_trends.return_value = {
            "topics": ["Test Topic"],
            "summary": "Test summary",
        }
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
            market_provider=mock_market,
        )
        
        assert sim._kalshi_client is mock_market

    def test_inject_user_pool_provider(self):
        """Test injecting custom user pool provider."""
        mock_user_pool = MagicMock(spec=UserPoolProviderABC)
        mock_user_pool.fetch_user_pool.return_value = [
            {"id": 0, "name": "MockUser", "personality_traits": ["test"]}
        ]
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
            user_pool_provider=mock_user_pool,
        )
        
        personas = sim._try_load_socioverse()
        
        mock_user_pool.fetch_user_pool.assert_called_once_with(count=1)
        assert personas[0]["name"] == "MockUser"

    def test_inject_all_providers(self):
        """Test injecting all providers together."""
        mock_llm = MagicMock(spec=LLMInterfaceABC)
        mock_llm.generate.return_value = "ACTION: HOLD\nCONTENT: Test"
        mock_llm.health_check.return_value = True
        
        mock_market = MagicMock(spec=MarketDataProviderABC)
        mock_market.get_trending_events.return_value = []
        mock_market.analyze_trends.return_value = {"topics": [], "summary": ""}
        
        mock_user_pool = MagicMock(spec=UserPoolProviderABC)
        mock_user_pool.fetch_user_pool.return_value = [
            {"id": 0, "name": "User1"},
            {"id": 1, "name": "User2"},
        ]
        
        sim = Simulation(
            days=1,
            agent_count=2,
            mock_llm=True,
            use_kalshi=True,
            llm_provider=mock_llm,
            market_provider=mock_market,
            user_pool_provider=mock_user_pool,
        )
        
        assert sim.llm is mock_llm
        assert sim._kalshi_client is mock_market
        assert sim._user_pool_provider is mock_user_pool

    def test_default_providers_created_when_not_injected(self):
        """Test default providers are created when not injected."""
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=False,
        )
        
        assert sim.llm is None
        assert sim._kalshi_client is None
        assert sim._user_pool_provider is None
        
        sim.setup()
        
        assert sim.llm is not None
        assert isinstance(sim.llm, LlamaInterface)

    def test_kalshi_client_created_when_use_kalshi_true(self):
        """Test KalshiClient is created when use_kalshi=True and not injected."""
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=True,
        )
        
        assert sim._kalshi_client is not None
        assert isinstance(sim._kalshi_client, KalshiClient)


class TestMockProviderBehavior:
    """Test that mock providers work correctly in simulation."""

    def test_mock_llm_used_in_agent_decision(self, sample_persona):
        """Test mock LLM is used when agent makes decision."""
        mock_llm = MagicMock(spec=LLMInterfaceABC)
        mock_llm.generate.return_value = "ACTION: TWEET\nCONTENT: Test tweet!"
        mock_llm.health_check.return_value = True
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=True,
            use_kalshi=False,
            llm_provider=mock_llm,
            custom_agents=[sample_persona],
        )
        sim.setup()
        
        assert len(sim.agents) == 1
        
        decision = sim.agents[0].decide(sim.llm)
        
        mock_llm.generate.assert_called()
        assert "TWEET" in decision or "Test tweet" in decision

    def test_mock_user_pool_populates_agents(self):
        """Test mock user pool provider populates agents correctly."""
        mock_user_pool = MagicMock(spec=UserPoolProviderABC)
        mock_user_pool.fetch_user_pool.return_value = [
            {
                "id": 0,
                "name": "InjectedUser1",
                "personality_traits": ["analytical"],
                "interests": ["markets"],
                "beliefs": {"risk_tolerance": "low"},
                "social": {"follower_count": 100, "influence_score": 0.5},
            },
            {
                "id": 1,
                "name": "InjectedUser2",
                "personality_traits": ["impulsive"],
                "interests": ["crypto"],
                "beliefs": {"risk_tolerance": "high"},
                "social": {"follower_count": 5000, "influence_score": 0.9},
            },
        ]
        
        sim = Simulation(
            days=1,
            agent_count=2,
            mock_llm=True,
            use_kalshi=True,
            user_pool_provider=mock_user_pool,
        )
        sim.setup()
        
        assert len(sim.agents) == 2
        assert sim.agents[0].name == "InjectedUser1"
        assert sim.agents[1].name == "InjectedUser2"
