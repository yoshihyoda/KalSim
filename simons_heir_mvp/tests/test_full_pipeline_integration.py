"""End-to-end integration tests for the full 7-layer pipeline.

Tests the complete flow from market stimulus through all layers
to the final agent decision, verifying that psychological state
correctly influences behavior.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.agent import Agent, MarketInfo, SocialMediaInfo
from src.prompt_builder import PromptBuilder
from src.core.behavior_engine import BehaviorEngine


class TestFullPipelineFlow:
    """Test the complete observe -> layers -> prompt -> decide pipeline."""

    def test_market_surge_triggers_fomo_increase(self, sample_persona):
        """Test that a market surge increases FOMO via the layer pipeline."""
        agent = Agent(agent_id=0, persona=sample_persona)
        initial_fomo = agent.state.neurobiological.fomo_level
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=300.0,
            price_change_pct=100.0,
            volume=100000000,
            trend="surging",
        )
        
        social_info = SocialMediaInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            trending_topics=["$GME", "squeeze", "moon"],
            sample_tweets=["To the moon!", "Diamond hands!"],
            sentiment_score=0.9,
        )
        
        agent.observe(market_info, social_info)
        
        assert agent.state.neurobiological.fomo_level > initial_fomo
        assert agent._last_layer_outputs.get("fomo_level", 0) > initial_fomo

    def test_market_surge_triggers_arousal_or_fomo(self, sample_persona):
        """Test that a market surge leads to increased arousal or FOMO."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=300.0,
            price_change_pct=100.0,
            volume=100000000,
            trend="surging",
        )
        
        agent.observe(market_info, None)
        
        assert (
            agent.state.emotional.arousal >= 0.5 or 
            agent.state.neurobiological.fomo_level > 0.5
        )

    def test_high_fomo_reflected_in_prompt(self, sample_persona):
        """Test that high FOMO state is reflected in the generated prompt."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {
            "fomo_level": 0.85,
            "stress_level": 0.3,
            "dominant_emotion": "excitement",
            "emotion_intensity": 0.7,
            "arousal": 0.8,
        }
        
        prompt = agent._build_decision_prompt()
        
        assert "FOMO" in prompt
        assert "URGENT" in prompt or "intense" in prompt

    def test_high_stress_reflected_in_prompt(self, sample_persona):
        """Test that high stress state is reflected in the generated prompt."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {
            "fomo_level": 0.3,
            "stress_level": 0.85,
            "dominant_emotion": "fear",
            "emotion_intensity": 0.7,
        }
        
        prompt = agent._build_decision_prompt()
        
        assert "stress" in prompt.lower()
        assert "caution" in prompt.lower()

    def test_full_pipeline_stimulus_to_decision(self, sample_persona, mock_llm_interface):
        """Test complete flow from market stimulus to agent decision."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=300.0,
            price_change_pct=100.0,
            volume=100000000,
            trend="surging",
        )
        
        social_info = SocialMediaInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            trending_topics=["$GME", "squeeze"],
            sample_tweets=["Hold!"],
            sentiment_score=0.9,
        )
        
        agent.observe(market_info, social_info)
        
        assert agent.state.neurobiological.fomo_level > 0
        
        decision = agent.decide(mock_llm_interface)
        
        mock_llm_interface.generate.assert_called_once()
        call_args = mock_llm_interface.generate.call_args
        prompt_used = call_args[0][0]
        
        assert "PSYCHOLOGICAL STATE" in prompt_used
        
        result = agent.act(decision)
        
        assert result.agent_id == 0
        assert result.action_type in ["TWEET", "HOLD", "LURK"]


class TestBehaviorEngineIntegration:
    """Test BehaviorEngine integration in Agent."""

    def test_agent_has_behavior_engine(self, sample_persona):
        """Test that agent has a BehaviorEngine instance."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        assert hasattr(agent, "behavior_engine")
        assert isinstance(agent.behavior_engine, BehaviorEngine)

    def test_behavior_engine_has_all_layers(self, sample_persona):
        """Test that BehaviorEngine has all 7 layers registered."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        assert len(agent.behavior_engine.pipeline.layers) == 7

    def test_layer_outputs_stored_after_observe(self, sample_persona):
        """Test that layer outputs are stored after observation."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=100.0,
            price_change_pct=10.0,
            volume=10000000,
            trend="rising",
        )
        
        agent.observe(market_info, None)
        
        assert agent._last_layer_outputs is not None
        assert len(agent._last_layer_outputs) > 0


class TestObserveAutoUpdatesLayers:
    """Test that observe() automatically updates layer states."""

    def test_observe_triggers_layer_update(self, sample_persona):
        """Test that observe automatically triggers layer state update."""
        agent = Agent(agent_id=0, persona=sample_persona)
        initial_fomo = agent.state.neurobiological.fomo_level
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=200.0,
            price_change_pct=50.0,
            volume=50000000,
            trend="surging",
        )
        
        agent.observe(market_info, None, auto_update_layers=True)
        
        assert agent.state.neurobiological.fomo_level != initial_fomo or agent._last_layer_outputs

    def test_observe_can_skip_layer_update(self, sample_persona):
        """Test that layer update can be skipped if needed."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=100.0,
            price_change_pct=10.0,
            volume=10000000,
            trend="rising",
        )
        
        initial_outputs = agent._last_layer_outputs.copy()
        
        agent.observe(market_info, None, auto_update_layers=False)
        
        assert len(agent.memory) > 0


class TestPromptBuilderIntegration:
    """Test PromptBuilder integration in Agent."""

    def test_decide_uses_prompt_builder(self, sample_persona, mock_llm_interface):
        """Test that decide() uses PromptBuilder for prompt generation."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {
            "fomo_level": 0.6,
            "dominant_emotion": "anticipation",
            "emotion_intensity": 0.5,
        }
        
        agent.decide(mock_llm_interface)
        
        call_args = mock_llm_interface.generate.call_args
        prompt = call_args[0][0]
        
        assert "CHARACTER PROFILE" in prompt
        assert "PSYCHOLOGICAL STATE" in prompt
        assert agent.name in prompt

    def test_prompt_reflects_identity_group(self, sample_persona, mock_llm_interface):
        """Test that prompt includes agent's identity group."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent.decide(mock_llm_interface)
        
        call_args = mock_llm_interface.generate.call_args
        prompt = call_args[0][0]
        
        assert agent.identity_group in prompt or "Identity Group" in prompt


class TestLayerOutputsAffectPrompt:
    """Test that different layer outputs produce different prompts."""

    def test_different_fomo_produces_different_prompt(self, sample_persona):
        """Test that different FOMO levels produce different prompts."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {"fomo_level": 0.1}
        low_fomo_prompt = agent._build_decision_prompt()
        
        agent._last_layer_outputs = {"fomo_level": 0.9}
        high_fomo_prompt = agent._build_decision_prompt()
        
        assert low_fomo_prompt != high_fomo_prompt
        assert "FOMO" in high_fomo_prompt
        assert "URGENT" in high_fomo_prompt

    def test_different_emotions_produce_different_prompts(self, sample_persona):
        """Test that different emotions produce different prompts."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {
            "dominant_emotion": "excitement",
            "emotion_intensity": 0.8,
        }
        excitement_prompt = agent._build_decision_prompt()
        
        agent._last_layer_outputs = {
            "dominant_emotion": "fear",
            "emotion_intensity": 0.8,
        }
        fear_prompt = agent._build_decision_prompt()
        
        assert excitement_prompt != fear_prompt
        assert "excited" in excitement_prompt.lower()
        assert "fear" in fear_prompt.lower()

    def test_social_pressure_affects_prompt(self, sample_persona):
        """Test that social pressure affects prompt content."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {"social_pressure": 0.1}
        low_pressure_prompt = agent._build_decision_prompt()
        
        agent._last_layer_outputs = {"social_pressure": 0.8}
        high_pressure_prompt = agent._build_decision_prompt()
        
        assert "social pressure" in high_pressure_prompt.lower()

    def test_herding_affects_prompt(self, sample_persona):
        """Test that herding detection affects prompt content."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        agent._last_layer_outputs = {"herding_detected": True}
        prompt = agent._build_decision_prompt()
        
        assert "herd" in prompt.lower()


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    def test_bull_market_scenario(self, sample_persona, mock_llm_interface):
        """Test agent behavior in a bull market scenario."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        for price, change, trend in [
            (50.0, 5.0, "rising"),
            (75.0, 25.0, "rising"),
            (120.0, 60.0, "surging"),
        ]:
            market_info = MarketInfo(
                timestamp=datetime(2024, 1, 1, 10, 0),
                stock_price=price,
                price_change_pct=change,
                volume=50000000,
                trend=trend,
            )
            agent.observe(market_info, None)
        
        assert agent.state.neurobiological.fomo_level > 0.3
        
        agent.decide(mock_llm_interface)
        call_args = mock_llm_interface.generate.call_args
        prompt = call_args[0][0]
        
        assert "PSYCHOLOGICAL STATE" in prompt

    def test_market_crash_scenario(self, sample_persona, mock_llm_interface):
        """Test agent behavior in a market crash scenario."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        market_info = MarketInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            stock_price=50.0,
            price_change_pct=-30.0,
            volume=100000000,
            trend="crashing",
        )
        
        agent.observe(market_info, None)
        
        agent.decide(mock_llm_interface)
        call_args = mock_llm_interface.generate.call_args
        prompt = call_args[0][0]
        
        assert "PSYCHOLOGICAL STATE" in prompt

    def test_viral_post_scenario(self, sample_persona, mock_llm_interface):
        """Test agent behavior after viral post exposure."""
        agent = Agent(agent_id=0, persona=sample_persona)
        
        social_info = SocialMediaInfo(
            timestamp=datetime(2024, 1, 1, 10, 0),
            trending_topics=["$GME", "viral", "squeeze"],
            sample_tweets=["VIRAL: Everyone is buying!"] * 10,
            sentiment_score=0.95,
        )
        
        agent._last_layer_outputs["viral_exposure"] = True
        agent._last_layer_outputs["social_pressure"] = 0.8
        
        agent.observe(None, social_info)
        
        prompt = agent._build_decision_prompt()
        
        assert "viral" in prompt.lower() or "social pressure" in prompt.lower()
