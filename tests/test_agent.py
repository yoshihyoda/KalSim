"""Tests for refactored Agent module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.agent import Agent, AgentState, MarketInfo, SocialMediaInfo


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_state_creation(self):
        """Test creating AgentState with all layer states."""
        state = AgentState()
        assert state.neurobiological is not None
        assert state.cognitive is not None
        assert state.emotional is not None

    def test_state_to_dict(self):
        """Test converting AgentState to dictionary."""
        state = AgentState()
        state_dict = state.to_dict()

        assert "fomo_level" in state_dict
        assert "valence" in state_dict
        assert "social_proof" in state_dict


class TestAgent:
    """Tests for refactored Agent class."""

    def test_agent_initialization(self, sample_persona):
        """Test initializing an Agent."""
        agent = Agent(agent_id=0, persona=sample_persona)
        assert agent.agent_id == 0
        assert agent.name == "TestUser"

    def test_agent_has_layer_state(self, sample_persona):
        """Test that agent has layer state."""
        agent = Agent(agent_id=0, persona=sample_persona)
        assert agent.state is not None

    def test_observe_updates_state(self, sample_persona, base_market_state):
        """Test that observe updates internal state."""
        agent = Agent(agent_id=0, persona=sample_persona)

        market_info = MarketInfo(
            timestamp=datetime.now(),
            stock_price=100.0,
            price_change_pct=50.0,
            volume=10000000,
            trend="surging",
        )

        agent.observe(market_info, None)

        assert len(agent.memory) > 0

    def test_observe_market_surge_increases_fomo(self, sample_persona):
        """Test that observing market surge increases FOMO."""
        agent = Agent(agent_id=0, persona=sample_persona)
        initial_fomo = agent.state.neurobiological.fomo_level

        market_info = MarketInfo(
            timestamp=datetime.now(),
            stock_price=300.0,
            price_change_pct=100.0,
            volume=50000000,
            trend="surging",
        )

        agent.observe(market_info, None)
        agent.update_layer_states(market_info, None)

        assert agent.state.neurobiological.fomo_level > initial_fomo

    def test_update_layer_states_pipeline(self, sample_persona):
        """Test that update_layer_states runs the full pipeline."""
        agent = Agent(agent_id=0, persona=sample_persona)

        market_info = MarketInfo(
            timestamp=datetime.now(),
            stock_price=200.0,
            price_change_pct=30.0,
            volume=20000000,
            trend="rising",
        )

        social_info = SocialMediaInfo(
            timestamp=datetime.now(),
            trending_topics=["$GME", "diamondhands"],
            sample_tweets=["Hold the line!"],
            sentiment_score=0.8,
        )

        agent.update_layer_states(market_info, social_info)

        assert agent.state.neurobiological.fomo_level > 0

    def test_build_prompt_includes_layer_context(self, sample_persona, mock_llm_interface):
        """Test that build prompt includes layer state context."""
        agent = Agent(agent_id=0, persona=sample_persona)

        agent.state.neurobiological.fomo_level = 0.8
        agent.state.emotional.dominant_emotion = "excitement"

        prompt = agent._build_decision_prompt()

        assert "CHARACTER PROFILE" in prompt or "CONTEXT" in prompt

    def test_decide_uses_llm(self, sample_persona, mock_llm_interface):
        """Test that decide method uses LLM interface."""
        agent = Agent(agent_id=0, persona=sample_persona)

        decision = agent.decide(mock_llm_interface)

        mock_llm_interface.generate.assert_called_once()
        assert "HOLD" in decision or "ACTION" in decision

    def test_act_returns_action_result(self, sample_persona):
        """Test that act returns ActionResult."""
        agent = Agent(agent_id=0, persona=sample_persona)

        decision = "ACTION: TWEET\nCONTENT: GME to the moon!"
        result = agent.act(decision)

        assert result.action_type == "TWEET"
        assert "moon" in result.content

    def test_get_layer_summary(self, sample_persona):
        """Test getting summary of layer states."""
        agent = Agent(agent_id=0, persona=sample_persona)

        summary = agent.get_layer_summary()

        assert isinstance(summary, str)

    def test_reset_layers(self, sample_persona):
        """Test resetting layer states."""
        agent = Agent(agent_id=0, persona=sample_persona)

        agent.state.neurobiological.fomo_level = 0.9
        agent.state.emotional.valence = 0.8

        agent.reset_layers()

        assert agent.state.neurobiological.fomo_level == 0.0

    def test_personality_affects_identity(self, sample_personas):
        """Test that personality traits affect identity assignment."""
        wsb_persona = sample_personas[0]
        cautious_persona = sample_personas[1]

        wsb_agent = Agent(agent_id=0, persona=wsb_persona)
        cautious_agent = Agent(agent_id=1, persona=cautious_persona)

        assert wsb_agent.identity_group != cautious_agent.identity_group
