"""Tests for PromptBuilder module."""

import pytest
from src.prompt_builder import PromptBuilder


class TestPromptBuilderInitialization:
    """Tests for PromptBuilder initialization."""

    def test_init_with_agent_state(self):
        """Test initializing with agent state."""
        agent_state = {"name": "TestUser", "identity_group": "WSB_APE"}
        builder = PromptBuilder(agent_state=agent_state)
        
        assert builder.agent_state == agent_state
        assert builder.layer_outputs == {}

    def test_init_with_layer_outputs(self):
        """Test initializing with layer outputs."""
        agent_state = {"name": "TestUser"}
        layer_outputs = {"fomo_level": 0.8, "stress_level": 0.3}
        
        builder = PromptBuilder(agent_state=agent_state, layer_outputs=layer_outputs)
        
        assert builder.layer_outputs == layer_outputs


class TestFomoContext:
    """Tests for FOMO context generation."""

    def test_high_fomo_generates_urgent_context(self):
        """Test high FOMO (>0.7) generates urgent language."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"fomo_level": 0.85}
        )
        
        context = builder._get_fomo_context()
        
        assert context is not None
        assert "URGENT" in context
        assert "intense FOMO" in context
        assert "0.8" in context  # Rounded value

    def test_moderate_fomo_generates_moderate_context(self):
        """Test moderate FOMO (0.4-0.7) generates appropriate language."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"fomo_level": 0.55}
        )
        
        context = builder._get_fomo_context()
        
        assert context is not None
        assert "moderate fear of missing out" in context
        assert "URGENT" not in context

    def test_low_fomo_returns_none(self):
        """Test low FOMO (<0.4) returns None."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"fomo_level": 0.2}
        )
        
        context = builder._get_fomo_context()
        
        assert context is None


class TestStressContext:
    """Tests for stress context generation."""

    def test_high_stress_generates_pressure_context(self):
        """Test high stress (>0.7) generates pressure language."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"stress_level": 0.85}
        )
        
        context = builder._get_stress_context()
        
        assert context is not None
        assert "highly stressed" in context
        assert "heart is racing" in context

    def test_moderate_stress_generates_tension_context(self):
        """Test moderate stress generates tension language."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"stress_level": 0.55}
        )
        
        context = builder._get_stress_context()
        
        assert context is not None
        assert "tension" in context

    def test_low_stress_returns_none(self):
        """Test low stress returns None."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"stress_level": 0.2}
        )
        
        context = builder._get_stress_context()
        
        assert context is None


class TestEmotionContext:
    """Tests for emotion context generation."""

    def test_excitement_emotion_context(self):
        """Test excitement emotion generates appropriate context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={
                "dominant_emotion": "excitement",
                "emotion_intensity": 0.8,
                "arousal": 0.75,
            }
        )
        
        context = builder._get_emotion_context()
        
        assert context is not None
        assert "excited and energized" in context
        assert "very intense" in context
        assert "ready to act" in context

    def test_fear_emotion_context(self):
        """Test fear emotion generates appropriate context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={
                "dominant_emotion": "fear",
                "emotion_intensity": 0.6,
            }
        )
        
        context = builder._get_emotion_context()
        
        assert context is not None
        assert "fearful and anxious" in context

    def test_neutral_emotion_returns_none(self):
        """Test neutral emotion returns None."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"dominant_emotion": "neutral"}
        )
        
        context = builder._get_emotion_context()
        
        assert context is None


class TestSocialContext:
    """Tests for social context generation."""

    def test_high_social_pressure_context(self):
        """Test high social pressure generates context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"social_pressure": 0.7}
        )
        
        context = builder._get_social_context()
        
        assert context is not None
        assert "social pressure" in context

    def test_herding_detected_context(self):
        """Test herding detection generates context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"herding_detected": True}
        )
        
        context = builder._get_social_context()
        
        assert context is not None
        assert "herd" in context

    def test_viral_exposure_context(self):
        """Test viral exposure generates context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"viral_exposure": True}
        )
        
        context = builder._get_social_context()
        
        assert context is not None
        assert "viral" in context

    def test_combined_social_factors(self):
        """Test multiple social factors combine in context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={
                "social_pressure": 0.7,
                "herding_detected": True,
                "viral_exposure": True,
            }
        )
        
        context = builder._get_social_context()
        
        assert context is not None
        assert "social pressure" in context
        assert "herd" in context
        assert "viral" in context


class TestIdentityContext:
    """Tests for identity context generation."""

    def test_strong_wsb_identity_context(self):
        """Test strong WSB identity generates context."""
        builder = PromptBuilder(
            agent_state={
                "name": "Test",
                "identity_state": {
                    "primary_group": "WSB_APE",
                    "group_identification": 0.8,
                }
            },
            layer_outputs={}
        )
        
        context = builder._get_identity_context()
        
        assert context is not None
        assert "WSB ape community" in context
        assert "diamond hands" in context

    def test_weak_identity_returns_none(self):
        """Test weak identity returns None."""
        builder = PromptBuilder(
            agent_state={
                "name": "Test",
                "identity_state": {
                    "primary_group": "WSB_APE",
                    "group_identification": 0.3,
                }
            },
            layer_outputs={}
        )
        
        context = builder._get_identity_context()
        
        assert context is None


class TestCognitiveContext:
    """Tests for cognitive bias context generation."""

    def test_high_social_proof_context(self):
        """Test high social proof generates context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"social_proof": 0.8}
        )
        
        context = builder._get_cognitive_context()
        
        assert context is not None
        assert "social proof" in context

    def test_high_loss_aversion_context(self):
        """Test high loss aversion generates context."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"loss_aversion": 0.85}
        )
        
        context = builder._get_cognitive_context()
        
        assert context is not None
        assert "loss-averse" in context


class TestBuildDecisionPrompt:
    """Tests for full prompt building."""

    def test_build_complete_prompt(self):
        """Test building a complete decision prompt."""
        agent_state = {
            "name": "DiamondHands",
            "personality_summary": "Risk-seeking, impulsive",
            "identity_group": "WSB_APE",
        }
        layer_outputs = {
            "fomo_level": 0.8,
            "stress_level": 0.3,
            "dominant_emotion": "excitement",
            "emotion_intensity": 0.7,
            "arousal": 0.8,
        }
        
        builder = PromptBuilder(agent_state, layer_outputs)
        prompt = builder.build_decision_prompt(
            market_topic="Bitcoin ETF",
            recent_context="Market up 10% today"
        )
        
        assert "DiamondHands" in prompt
        assert "Bitcoin ETF" in prompt
        assert "PSYCHOLOGICAL STATE" in prompt
        assert "FOMO" in prompt
        assert "ACTION:" in prompt
        assert "TWEET/HOLD/LURK" in prompt

    def test_prompt_with_no_significant_state(self):
        """Test prompt when no significant psychological state."""
        agent_state = {"name": "CalmTrader"}
        layer_outputs = {
            "fomo_level": 0.1,
            "stress_level": 0.1,
            "dominant_emotion": "neutral",
        }
        
        builder = PromptBuilder(agent_state, layer_outputs)
        prompt = builder.build_decision_prompt()
        
        assert "calm and analytical" in prompt

    def test_prompt_action_guidance_high_fomo(self):
        """Test action guidance with high FOMO and arousal."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={
                "fomo_level": 0.85,
                "arousal": 0.8,
            }
        )
        
        guidance = builder._build_action_guidance()
        
        assert "strong urge to engage" in guidance

    def test_prompt_action_guidance_high_stress(self):
        """Test action guidance with high stress."""
        builder = PromptBuilder(
            agent_state={"name": "Test"},
            layer_outputs={"stress_level": 0.85}
        )
        
        guidance = builder._build_action_guidance()
        
        assert "caution" in guidance


class TestGetStateSummary:
    """Tests for state summary utility."""

    def test_get_state_summary(self):
        """Test getting state summary."""
        layer_outputs = {
            "fomo_level": 0.7,
            "stress_level": 0.5,
            "dominant_emotion": "excitement",
            "emotion_intensity": 0.8,
            "social_pressure": 0.6,
            "herding_detected": True,
            "viral_exposure": False,
        }
        
        builder = PromptBuilder(agent_state={}, layer_outputs=layer_outputs)
        summary = builder.get_state_summary()
        
        assert summary["fomo_level"] == 0.7
        assert summary["dominant_emotion"] == "excitement"
        assert summary["herding_detected"] is True

    def test_get_state_summary_defaults(self):
        """Test state summary with missing values uses defaults."""
        builder = PromptBuilder(agent_state={}, layer_outputs={})
        summary = builder.get_state_summary()
        
        assert summary["fomo_level"] == 0.0
        assert summary["dominant_emotion"] == "neutral"
        assert summary["herding_detected"] is False
