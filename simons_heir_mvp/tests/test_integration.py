"""Integration tests for the 7-layer model simulation."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.agent import Agent, AgentState, MarketInfo, SocialMediaInfo
from src.core.social_environment import SocialEnvironment
from src.core.user_engine import UserEngine
from src.core.scenario_engine import ScenarioEngine
from src.core.behavior_engine import BehaviorEngine
from src.layers.layer1_neurobiology import NeurobiologyModule
from src.layers.layer2_cognition import CognitionModule
from src.layers.layer3_emotion import EmotionModule
from src.layers.layer4_social_interaction import SocialInteractionModule
from src.layers.layer5_collective_identity import IdentityModule, IdentityGroup
from src.layers.layer6_network_structure import NetworkStructureModule
from src.layers.layer7_market_structure import MarketStructureModule


class TestLayerIntegration:
    """Test integration between multiple layers."""

    def test_market_surge_triggers_fomo_and_excitement(self):
        """Test that market surge flows through L7 -> L1 -> L3."""
        market_module = MarketStructureModule()
        neuro_module = NeurobiologyModule()
        emotion_module = EmotionModule()

        market_state = {
            "market": {"price": 300.0, "volume": 50000000, "short_interest": 140.0},
            "agent_actions": [{"action": "BUY", "agent_id": i} for i in range(30)],
        }
        market_output = market_module.process(market_state)

        assert market_output["price_pressure"] > 0
        assert market_output["short_squeeze_pressure"] > 0

        neuro_state = {
            "agent": {"neuro_state": {"fomo_level": 0.2}},
            "market": {"price_change_pct": 50.0, "trend": "surging"},
            "social": {"sentiment": 0.8},
        }
        neuro_output = neuro_module.process(neuro_state)

        assert neuro_output["fomo_level"] > 0.2

        emotion_state = {
            "agent": {"emotion": {"valence": 0.2, "arousal": 0.3}},
            "stimulus": {"type": "market_surge", "intensity": 0.8},
        }
        emotion_output = emotion_module.process(emotion_state)

        assert emotion_output["valence"] > 0.2
        assert emotion_output["arousal"] > 0.3

    def test_viral_post_triggers_emotion_contagion(self):
        """Test that viral post flows through L6 -> L4."""
        network_module = NetworkStructureModule(viral_threshold=100)
        social_module = SocialInteractionModule()

        for i in range(150):
            network_module.platform.upvote("post_001", voter_id=i)

        network_state = {
            "social": {
                "viral_posts": [{"id": "post_001", "upvotes": 5000, "sentiment": 0.9}],
            },
            "agent": {"id": 0},
        }
        network_output = network_module.process(network_state)

        assert network_output["viral_exposure"] is True
        assert network_output["virality_intensity"] > 0

        social_state = {
            "agent": {
                "id": 0,
                "emotion": {"valence": 0.2, "arousal": 0.3},
            },
            "social": {
                "neighbors": [
                    {"id": i, "emotion": {"valence": 0.9, "arousal": 0.8}}
                    for i in range(1, 6)
                ],
                "connection_strengths": {i: 0.8 for i in range(1, 6)},
                "consensus_strength": 0.9,
            },
        }
        social_output = social_module.process(social_state)

        assert social_output["emotion_received"]["valence"] > 0.2
        assert social_output["social_pressure"] > 0

    def test_identity_affects_trust_filtering(self):
        """Test that identity (L5) affects trust in messages."""
        identity_module = IdentityModule()

        wsb_persona = {
            "personality_traits": ["impulsive", "community-focused"],
            "interests": ["wsb", "memes", "crypto"],
            "beliefs": {"risk_tolerance": "high"},
        }

        wsb_identity = identity_module.assign_identity(wsb_persona)
        assert wsb_identity.primary_group == IdentityGroup.WSB_APE

        in_group_trust = identity_module.adjust_trust(
            wsb_identity, IdentityGroup.WSB_APE, base_trust=0.5
        )
        out_group_trust = identity_module.adjust_trust(
            wsb_identity, IdentityGroup.INSTITUTIONAL, base_trust=0.5
        )

        assert in_group_trust > out_group_trust

    def test_cognitive_bias_with_social_proof(self):
        """Test that social proof (L2) increases with peer count."""
        cognition_module = CognitionModule()

        state_many_peers = {
            "agent": {"beliefs": {"gme_bullish": 0.4}},
            "social": {
                "consensus_view": "bullish",
                "consensus_strength": 0.8,
                "peer_count": 100,
            },
            "market": {},
            "information": {},
        }

        state_few_peers = {
            "agent": {"beliefs": {"gme_bullish": 0.4}},
            "social": {
                "consensus_view": "bullish",
                "consensus_strength": 0.8,
                "peer_count": 5,
            },
            "market": {},
            "information": {},
        }

        result_many = cognition_module.process(state_many_peers)
        cognition_module.reset()
        result_few = cognition_module.process(state_few_peers)

        assert result_many["adjusted_belief"] > result_few["adjusted_belief"]


class TestAgentIntegration:
    """Test Agent class with 7-layer integration."""

    def test_agent_full_pipeline(self, sample_persona, mock_llm_interface):
        """Test agent processes through full observe -> update -> decide -> act pipeline."""
        agent = Agent(agent_id=0, persona=sample_persona)

        market_info = MarketInfo(
            timestamp=datetime(2021, 1, 27, 14, 0),
            stock_price=350.0,
            price_change_pct=100.0,
            volume=100000000,
            trend="surging",
        )

        social_info = SocialMediaInfo(
            timestamp=datetime(2021, 1, 27, 14, 0),
            trending_topics=["$GME", "diamondhands", "tothemoon"],
            sample_tweets=["HOLD THE LINE!", "Diamond hands forever!"],
            sentiment_score=0.9,
        )

        agent.observe(market_info, social_info)
        agent.update_layer_states(market_info, social_info)

        assert agent.state.neurobiological.fomo_level > 0

        decision = agent.decide(mock_llm_interface)
        assert decision is not None

        result = agent.act(decision)
        assert result.agent_id == 0
        assert "layer_state" in result.metadata

    def test_agent_identity_assignment(self, sample_personas):
        """Test that different personas get different identities."""
        agents = [
            Agent(agent_id=i, persona=p)
            for i, p in enumerate(sample_personas)
        ]

        identity_groups = [a.identity_group for a in agents]

        assert "WSB_APE" in identity_groups


class TestCoreComponentIntegration:
    """Test core component integration."""

    def test_social_environment_with_scenario_engine(self, sample_network_edges):
        """Test SocialEnvironment works with ScenarioEngine."""
        env = SocialEnvironment()
        scenario = ScenarioEngine()
        scenario.build_network(edges=sample_network_edges)

        env.update_state(price=100.0, trend="rising")
        env.advance_time(hours=1)

        assert env.current_state.price == 100.0
        assert scenario.network.number_of_nodes() == 5

    def test_user_engine_loads_and_categorizes(self, sample_personas):
        """Test UserEngine loads personas and assigns demographics."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        assert len(engine.personas) == 3
        
        for persona_id in engine.get_all_ids():
            label = engine.get_demographic_label(persona_id)
            assert label is not None

    def test_behavior_engine_pipeline(self):
        """Test BehaviorEngine orchestrates layer pipeline."""
        engine = BehaviorEngine()

        neuro = NeurobiologyModule()
        emotion = EmotionModule()

        engine.register_layer(neuro)
        engine.register_layer(emotion)

        result = engine.process(
            agent_state={"neuro_state": {}},
            market_state={"price_change_pct": 30.0, "trend": "rising"},
            social_state={"sentiment": 0.7},
        )

        assert "fomo_level" in result or "agent" in result


class TestEndToEndScenario:
    """End-to-end scenario tests."""

    def test_gamestop_squeeze_scenario(self, sample_personas, mock_llm_interface):
        """Test full GME squeeze scenario simulation step."""
        env = SocialEnvironment()
        user_engine = UserEngine()
        user_engine.load_personas(personas=sample_personas)

        agents = [
            Agent(agent_id=i, persona=p)
            for i, p in enumerate(sample_personas)
        ]

        env.update_state(price=50.0, volume=20000000, trend="rising")
        env.advance_time(hours=1)
        env.update_state(price=100.0, volume=50000000, trend="surging")

        market_info = MarketInfo(
            timestamp=env.current_state.timestamp,
            stock_price=env.current_state.price,
            price_change_pct=100.0,
            volume=env.current_state.volume,
            trend=env.current_state.trend,
        )

        social_info = SocialMediaInfo(
            timestamp=env.current_state.timestamp,
            trending_topics=["$GME", "WSB", "squeeze"],
            sample_tweets=["To the moon!", "Diamond hands!"],
            sentiment_score=0.85,
        )

        actions = []
        for agent in agents:
            agent.observe(market_info, social_info)
            agent.update_layer_states(market_info, social_info)
            decision = agent.decide(mock_llm_interface)
            result = agent.act(decision)
            actions.append(result)

        assert len(actions) == len(agents)

        for agent in agents:
            assert agent.state.neurobiological.fomo_level >= 0

        action_types = [a.action_type for a in actions]
        assert len(action_types) == 3
