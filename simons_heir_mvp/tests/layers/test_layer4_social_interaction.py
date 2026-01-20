"""Tests for Layer 4: Social Interaction module."""

import pytest

from src.layers.layer4_social_interaction import (
    SocialInteractionModule,
    EmotionContagion,
    SocialInfluence,
)


class TestEmotionContagion:
    """Tests for EmotionContagion class."""

    def test_contagion_initialization(self):
        """Test initializing EmotionContagion."""
        contagion = EmotionContagion()
        assert contagion is not None
        assert contagion.susceptibility == 0.5

    def test_propagate_emotion_basic(self):
        """Test basic emotion propagation between agents."""
        contagion = EmotionContagion(susceptibility=0.5)

        source_emotion = {"valence": 0.8, "arousal": 0.7}
        target_emotion = {"valence": 0.2, "arousal": 0.3}
        connection_strength = 0.8

        result = contagion.propagate(
            source_emotion, target_emotion, connection_strength
        )

        assert result["valence"] > target_emotion["valence"]
        assert result["arousal"] > target_emotion["arousal"]

    def test_propagate_negative_emotion(self):
        """Test propagation of negative emotion."""
        contagion = EmotionContagion(susceptibility=0.6)

        source_emotion = {"valence": -0.7, "arousal": 0.9}
        target_emotion = {"valence": 0.3, "arousal": 0.2}
        connection_strength = 0.7

        result = contagion.propagate(
            source_emotion, target_emotion, connection_strength
        )

        assert result["valence"] < target_emotion["valence"]

    def test_weak_connection_reduces_contagion(self):
        """Test that weak connections reduce contagion."""
        contagion = EmotionContagion(susceptibility=0.5)

        source_emotion = {"valence": 0.9, "arousal": 0.8}
        target_emotion = {"valence": 0.1, "arousal": 0.1}

        strong_result = contagion.propagate(
            source_emotion, target_emotion, connection_strength=0.9
        )
        weak_result = contagion.propagate(
            source_emotion, target_emotion, connection_strength=0.2
        )

        assert strong_result["valence"] > weak_result["valence"]


class TestSocialInfluence:
    """Tests for SocialInfluence class."""

    def test_influence_initialization(self):
        """Test initializing SocialInfluence."""
        influence = SocialInfluence()
        assert influence is not None

    def test_calculate_influence_high_influence_agent(self):
        """Test influence calculation for high-influence agent."""
        influence = SocialInfluence()

        source = {"influence_score": 0.9, "follower_count": 10000}
        target = {"influence_score": 0.3, "follower_count": 100}

        result = influence.calculate(source, target)

        assert result > 0.5

    def test_calculate_influence_low_influence_agent(self):
        """Test influence calculation for low-influence agent."""
        influence = SocialInfluence()

        source = {"influence_score": 0.1, "follower_count": 50}
        target = {"influence_score": 0.8, "follower_count": 5000}

        result = influence.calculate(source, target)

        assert result < 0.3


class TestSocialInteractionModule:
    """Tests for SocialInteractionModule class."""

    def test_module_initialization(self):
        """Test initializing SocialInteractionModule."""
        module = SocialInteractionModule()
        assert module.name == "social_interaction"

    def test_process_with_network_neighbors(self):
        """Test processing with network neighbor emotions."""
        module = SocialInteractionModule()

        state = {
            "agent": {
                "id": 0,
                "emotion": {"valence": 0.3, "arousal": 0.3},
            },
            "social": {
                "neighbors": [
                    {"id": 1, "emotion": {"valence": 0.8, "arousal": 0.7}},
                    {"id": 2, "emotion": {"valence": 0.9, "arousal": 0.8}},
                ],
                "connection_strengths": {1: 0.8, 2: 0.6},
            },
        }

        result = module.process(state)

        assert "emotion_received" in result
        assert result["emotion_received"]["valence"] > 0.3

    def test_process_calculates_social_pressure(self):
        """Test that processing calculates social pressure."""
        module = SocialInteractionModule()

        state = {
            "agent": {"id": 0, "emotion": {"valence": 0.5, "arousal": 0.5}},
            "social": {
                "neighbors": [
                    {"id": i, "emotion": {"valence": 0.9, "arousal": 0.8}}
                    for i in range(1, 6)
                ],
                "consensus_action": "BUY",
                "consensus_strength": 0.9,
            },
        }

        result = module.process(state)

        assert "social_pressure" in result
        assert result["social_pressure"] > 0

    def test_aggregate_neighbor_emotions(self):
        """Test aggregating emotions from multiple neighbors."""
        module = SocialInteractionModule()

        neighbors = [
            {"emotion": {"valence": 0.8, "arousal": 0.6}},
            {"emotion": {"valence": 0.6, "arousal": 0.7}},
            {"emotion": {"valence": 0.9, "arousal": 0.5}},
        ]
        weights = [0.5, 0.3, 0.2]

        result = module.aggregate_emotions(neighbors, weights)

        assert 0.6 < result["valence"] < 0.9
        assert 0.5 < result["arousal"] < 0.7

    def test_herding_behavior_detection(self):
        """Test detection of herding behavior."""
        module = SocialInteractionModule()

        state = {
            "agent": {"id": 0},
            "social": {
                "neighbors": [
                    {"id": i, "action": "BUY"} for i in range(1, 11)
                ],
            },
        }

        result = module.process(state)

        assert result.get("herding_detected", False) is True

    def test_no_herding_with_mixed_actions(self):
        """Test no herding with mixed neighbor actions."""
        module = SocialInteractionModule()

        state = {
            "agent": {"id": 0},
            "social": {
                "neighbors": [
                    {"id": 1, "action": "BUY"},
                    {"id": 2, "action": "SELL"},
                    {"id": 3, "action": "HOLD"},
                    {"id": 4, "action": "BUY"},
                    {"id": 5, "action": "SELL"},
                ],
            },
        }

        result = module.process(state)

        assert result.get("herding_detected", False) is False

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = SocialInteractionModule()
        module._last_social_pressure = 0.7

        summary = module.get_state_summary()

        assert "pressure" in summary.lower() or "0.7" in summary

    def test_reset(self):
        """Test resetting module state."""
        module = SocialInteractionModule()
        module._last_social_pressure = 0.8
        module._last_emotion_received = {"valence": 0.9}

        module.reset()

        assert module._last_social_pressure == 0.0
        assert module._last_emotion_received is None
