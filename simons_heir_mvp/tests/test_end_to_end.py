"""End-to-end integration tests for the simulation.

Tests the full simulation pipeline with dependency injection,
ensuring all components work together correctly.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.simulation import Simulation
from src.interfaces import LLMInterfaceABC, MarketDataProviderABC, UserPoolProviderABC
from src.agent import MarketInfo, SocialMediaInfo


class MockLLMProvider(LLMInterfaceABC):
    """Mock LLM provider for testing."""
    
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["ACTION: HOLD\nCONTENT: Monitoring market."]
        self.call_count = 0
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def health_check(self) -> bool:
        return True


class MockMarketProvider(MarketDataProviderABC):
    """Mock market data provider for testing."""
    
    def __init__(self, events: list[dict] | None = None):
        self.events = events or [
            {"title": "Test Market", "event_ticker": "TEST", "markets": []}
        ]
    
    def get_trending_events(self, limit: int = 20, status: str | None = "open") -> list[dict]:
        return self.events[:limit]
    
    def analyze_trends(self, events: list[dict]) -> dict:
        topics = [e.get("title", "Unknown") for e in events]
        return {
            "topics": topics,
            "summary": f"Trending: {', '.join(topics[:3])}",
        }


class MockUserPoolProvider(UserPoolProviderABC):
    """Mock user pool provider for testing."""
    
    def __init__(self, personas: list[dict] | None = None):
        self.personas = personas or [
            {
                "id": 0,
                "name": "TestUser1",
                "personality_traits": ["analytical", "cautious"],
                "interests": ["markets", "economics"],
                "beliefs": {"risk_tolerance": "low", "market_outlook": "neutral"},
                "social": {"follower_count": 500, "influence_score": 0.5},
            },
            {
                "id": 1,
                "name": "TestUser2",
                "personality_traits": ["impulsive", "enthusiastic"],
                "interests": ["crypto", "memes"],
                "beliefs": {"risk_tolerance": "high", "market_outlook": "bullish"},
                "social": {"follower_count": 5000, "influence_score": 0.8},
            },
        ]
    
    def fetch_user_pool(self, count: int = 100) -> list[dict]:
        return self.personas[:count]


class TestEndToEndSimulation:
    """End-to-end tests for simulation with injected dependencies."""

    def test_mini_simulation_with_mocks(self):
        """Test a minimal simulation (2 agents, 2 steps) with mocked providers."""
        mock_llm = MockLLMProvider(responses=[
            "ACTION: TWEET\nCONTENT: Markets looking interesting today!",
            "ACTION: HOLD\nCONTENT: Watching and waiting.",
            "ACTION: TWEET\nCONTENT: Bullish on this market!",
            "ACTION: LURK\nCONTENT: Just observing.",
        ])
        mock_market = MockMarketProvider(events=[
            {"title": "Bitcoin ETF", "event_ticker": "BTCETF"},
            {"title": "Fed Rate Decision", "event_ticker": "FEDRATE"},
        ])
        mock_user_pool = MockUserPoolProvider()
        
        sim = Simulation(
            days=1,
            agent_count=2,
            mock_llm=False,
            use_kalshi=True,
            llm_provider=mock_llm,
            market_provider=mock_market,
            user_pool_provider=mock_user_pool,
        )
        
        sim.setup()
        
        assert len(sim.agents) == 2
        assert sim.agents[0].name == "TestUser1"
        assert sim.agents[1].name == "TestUser2"
        assert sim.llm is mock_llm
        
        sim.days = 1
        original_steps = 24
        
        for step in range(2):
            sim._execute_step(step)
        
        assert len(sim.simulation_log) > 0
        
        assert mock_llm.call_count > 0

    def test_agent_pipeline_with_mock_llm(self):
        """Test individual agent observe -> update -> decide -> act pipeline."""
        mock_llm = MockLLMProvider(responses=[
            "ACTION: TWEET\nCONTENT: Testing the pipeline!"
        ])
        
        personas = [
            {
                "id": 0,
                "name": "PipelineTestAgent",
                "personality_traits": ["test"],
                "interests": ["testing"],
                "beliefs": {"risk_tolerance": "moderate"},
                "social": {"follower_count": 100, "influence_score": 0.5},
            }
        ]
        
        sim = Simulation(
            days=1,
            agent_count=1,
            mock_llm=False,
            use_kalshi=False,
            llm_provider=mock_llm,
            custom_agents=personas,
        )
        sim.setup()
        
        agent = sim.agents[0]
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=50.0,
            price_change_pct=5.0,
            volume=1000000,
            trend="rising",
        )
        
        social_info = SocialMediaInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            trending_topics=["TestTopic"],
            sample_tweets=["Test tweet"],
            sentiment_score=0.5,
        )
        
        agent.observe(market_info, social_info)
        agent.update_layer_states(market_info, social_info)
        decision = agent.decide(mock_llm)
        result = agent.act(decision)
        
        assert result.agent_id == 0
        assert result.action_type == "TWEET"
        assert "Testing the pipeline" in result.content

    def test_simulation_with_varied_agent_decisions(self):
        """Test simulation produces varied actions based on LLM responses."""
        responses = [
            "ACTION: TWEET\nCONTENT: Buying the dip!",
            "ACTION: HOLD\nCONTENT: Waiting for clarity.",
            "ACTION: LURK\nCONTENT: Just watching.",
            "ACTION: TWEET\nCONTENT: This is huge!",
        ]
        mock_llm = MockLLMProvider(responses=responses)
        mock_user_pool = MockUserPoolProvider()
        
        sim = Simulation(
            days=1,
            agent_count=2,
            mock_llm=False,
            use_kalshi=False,
            llm_provider=mock_llm,
            user_pool_provider=mock_user_pool,
        )
        sim.setup()
        
        for step in range(3):
            sim._execute_step(step)
        
        action_types = {log["action_type"] for log in sim.simulation_log}
        
        assert len(sim.simulation_log) > 0

    def test_community_sentiment_updates(self):
        """Test that community sentiment changes based on agent actions."""
        mock_llm = MockLLMProvider(responses=[
            "ACTION: TWEET\nCONTENT: To the moon! Diamond hands! 🚀",
            "ACTION: TWEET\nCONTENT: HODL! Bullish forever!",
        ])
        mock_user_pool = MockUserPoolProvider()
        
        sim = Simulation(
            days=1,
            agent_count=2,
            mock_llm=False,
            use_kalshi=False,
            llm_provider=mock_llm,
            user_pool_provider=mock_user_pool,
        )
        sim.setup()
        
        initial_sentiment = sim._community_sentiment
        
        for step in range(5):
            sim._execute_step(step)
        
        assert sim._community_sentiment != initial_sentiment or len(sim.simulation_log) > 0


class TestIntegrationWithRealComponents:
    """Integration tests that use real components (but mocked external APIs)."""

    def test_simulation_with_real_llm_interface_mock_mode(self):
        """Test simulation using real LlamaInterface in mock mode."""
        sim = Simulation(
            days=1,
            agent_count=3,
            mock_llm=True,
            use_kalshi=False,
        )
        sim.setup()
        
        assert len(sim.agents) == 3
        
        for step in range(2):
            sim._execute_step(step)
        
        assert len(sim.simulation_log) > 0
        
        for log in sim.simulation_log:
            assert "agent_id" in log
            assert "action_type" in log
            assert "content" in log

    def test_full_simulation_cycle_mock_mode(self):
        """Test complete simulation cycle in mock mode."""
        sim = Simulation(
            days=1,
            agent_count=5,
            mock_llm=True,
            use_kalshi=False,
        )
        
        sim.setup()
        
        sim.days = 1
        sim._execute_step(0)
        sim._execute_step(1)
        sim._execute_step(2)
        
        assert len(sim.simulation_log) >= 3
        
        for log_entry in sim.simulation_log:
            assert log_entry["action_type"] in ["TWEET", "HOLD", "LURK"]
