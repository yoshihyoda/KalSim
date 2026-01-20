"""Tests for Layer 2: Cognition module."""

import pytest

from src.layers.layer2_cognition import CognitionModule, CognitiveBiases


class TestCognitiveBiases:
    """Tests for CognitiveBiases dataclass."""

    def test_biases_creation(self):
        """Test creating CognitiveBiases."""
        biases = CognitiveBiases(
            social_proof=0.8,
            confirmation_bias=0.6,
            anchoring=0.5,
            loss_aversion=0.7,
        )
        assert biases.social_proof == 0.8
        assert biases.confirmation_bias == 0.6

    def test_default_values(self):
        """Test default bias values."""
        biases = CognitiveBiases()
        assert 0 <= biases.social_proof <= 1
        assert 0 <= biases.confirmation_bias <= 1


class TestCognitionModule:
    """Tests for CognitionModule class."""

    def test_module_initialization(self):
        """Test initializing CognitionModule."""
        module = CognitionModule()
        assert module.name == "cognition"

    def test_social_proof_bias_increases_agreement(self):
        """Test that social proof increases agreement with popular views."""
        module = CognitionModule()

        state = {
            "agent": {
                "beliefs": {"gme_bullish": 0.4},
            },
            "social": {
                "consensus_view": "bullish",
                "consensus_strength": 0.9,
                "peer_count": 50,
            },
        }

        result = module.process(state)

        assert result["adjusted_belief"] > 0.4
        assert "social_proof" in result.get("active_biases", [])

    def test_social_proof_weak_with_few_peers(self):
        """Test that social proof is weak with few peers."""
        module = CognitionModule()

        state_many = {
            "agent": {"beliefs": {"gme_bullish": 0.4}},
            "social": {"consensus_view": "bullish", "consensus_strength": 0.8, "peer_count": 100},
        }

        state_few = {
            "agent": {"beliefs": {"gme_bullish": 0.4}},
            "social": {"consensus_view": "bullish", "consensus_strength": 0.8, "peer_count": 3},
        }

        result_many = module.process(state_many)
        result_few = module.process(state_few)

        assert result_many["adjusted_belief"] > result_few["adjusted_belief"]

    def test_confirmation_bias_filters_information(self):
        """Test that confirmation bias filters contrary information."""
        module = CognitionModule()

        state = {
            "agent": {
                "beliefs": {"gme_bullish": 0.9},
            },
            "information": {
                "content": "GME is overvalued",
                "sentiment": -0.8,
                "contrary_to_belief": True,
            },
        }

        result = module.process(state)

        assert result.get("information_acceptance", 1.0) < 1.0

    def test_anchoring_bias_on_initial_price(self):
        """Test anchoring bias on initial price."""
        module = CognitionModule()

        state = {
            "agent": {
                "anchor_price": 20.0,
            },
            "market": {
                "current_price": 300.0,
            },
        }

        result = module.process(state)

        assert result.get("anchoring_effect", 0) > 0
        assert result.get("perceived_overvaluation", 0) > 0

    def test_loss_aversion_affects_selling(self):
        """Test that loss aversion affects selling decisions."""
        module = CognitionModule()

        state_loss = {
            "agent": {
                "entry_price": 100.0,
            },
            "market": {
                "current_price": 80.0,
            },
        }

        state_gain = {
            "agent": {
                "entry_price": 100.0,
            },
            "market": {
                "current_price": 120.0,
            },
        }

        result_loss = module.process(state_loss)
        result_gain = module.process(state_gain)

        assert result_loss.get("sell_reluctance", 0) > result_gain.get("sell_reluctance", 0)

    def test_bandwagon_effect(self):
        """Test bandwagon effect with rapid adoption."""
        module = CognitionModule()

        state = {
            "agent": {"initial_interest": 0.3},
            "social": {
                "adoption_rate": 0.8,
                "trend_velocity": 0.9,
            },
        }

        result = module.process(state)

        assert result.get("bandwagon_effect", 0) > 0

    def test_calculate_bias_strength(self):
        """Test calculating overall bias strength."""
        module = CognitionModule()

        biases = CognitiveBiases(
            social_proof=0.8,
            confirmation_bias=0.7,
            anchoring=0.5,
            loss_aversion=0.6,
        )

        strength = module.calculate_bias_strength(biases)

        assert 0 <= strength <= 1

    def test_identify_active_biases(self):
        """Test identifying which biases are active."""
        module = CognitionModule()

        state = {
            "agent": {"beliefs": {"gme_bullish": 0.8}},
            "social": {"consensus_strength": 0.9, "peer_count": 50},
            "information": {"contrary_to_belief": True},
        }

        result = module.process(state)
        active = result.get("active_biases", [])

        assert len(active) > 0

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = CognitionModule()
        module._current_biases = CognitiveBiases(social_proof=0.8)
        module._active_biases = ["social_proof"]

        summary = module.get_state_summary()

        assert "social_proof" in summary.lower() or "bias" in summary.lower()

    def test_reset(self):
        """Test resetting module state."""
        module = CognitionModule()
        module._current_biases = CognitiveBiases(social_proof=0.9)
        module._active_biases = ["social_proof", "anchoring"]

        module.reset()

        assert module._active_biases == []
