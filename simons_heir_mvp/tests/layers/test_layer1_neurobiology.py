"""Tests for Layer 1: Neurobiology module."""

import pytest

from src.layers.layer1_neurobiology import (
    NeurobiologyModule,
    NeurobiologicalState,
)


class TestNeurobiologicalState:
    """Tests for NeurobiologicalState dataclass."""

    def test_state_creation(self):
        """Test creating NeurobiologicalState."""
        state = NeurobiologicalState(
            fomo_level=0.8,
            dopamine_response=0.7,
            stress_level=0.3,
            reward_sensitivity=0.6,
        )
        assert state.fomo_level == 0.8
        assert state.dopamine_response == 0.7

    def test_default_values(self):
        """Test default neurobiological state values."""
        state = NeurobiologicalState()
        assert state.fomo_level == 0.0
        assert state.dopamine_response == 0.5


class TestNeurobiologyModule:
    """Tests for NeurobiologyModule class."""

    def test_module_initialization(self):
        """Test initializing NeurobiologyModule."""
        module = NeurobiologyModule()
        assert module.name == "neurobiology"

    def test_market_surge_increases_fomo(self):
        """Test that market surge increases FOMO level."""
        module = NeurobiologyModule()

        state = {
            "agent": {
                "neuro_state": {"fomo_level": 0.2, "dopamine_response": 0.5},
            },
            "market": {
                "price_change_pct": 50.0,
                "trend": "surging",
            },
        }

        result = module.process(state)

        assert result["fomo_level"] > 0.2

    def test_market_crash_increases_stress(self):
        """Test that market crash increases stress."""
        module = NeurobiologyModule()

        state = {
            "agent": {
                "neuro_state": {"stress_level": 0.2},
            },
            "market": {
                "price_change_pct": -30.0,
                "trend": "crashing",
            },
        }

        result = module.process(state)

        assert result["stress_level"] > 0.2

    def test_dopamine_response_to_gains(self):
        """Test dopamine response to unrealized gains."""
        module = NeurobiologyModule()

        state = {
            "agent": {
                "neuro_state": {"dopamine_response": 0.5},
                "portfolio": {"unrealized_pnl_pct": 100.0},
            },
            "market": {"trend": "rising"},
        }

        result = module.process(state)

        assert result["dopamine_response"] > 0.5

    def test_dopamine_diminishes_with_repeated_gains(self):
        """Test that dopamine response diminishes with repeated exposure."""
        module = NeurobiologyModule()

        initial_state = {
            "agent": {
                "neuro_state": {"dopamine_response": 0.5, "habituation": 0.0},
            },
            "market": {"price_change_pct": 20.0, "trend": "rising"},
        }

        result1 = module.process(initial_state)
        dopamine1 = result1["dopamine_response"]

        habituated_state = {
            "agent": {
                "neuro_state": {
                    "dopamine_response": dopamine1,
                    "habituation": result1.get("habituation", 0.3),
                },
            },
            "market": {"price_change_pct": 20.0, "trend": "rising"},
        }

        result2 = module.process(habituated_state)

        assert result2["dopamine_response"] <= dopamine1 + 0.1

    def test_fomo_triggers_urgency(self):
        """Test that high FOMO triggers urgency."""
        module = NeurobiologyModule()

        state = {
            "agent": {
                "neuro_state": {"fomo_level": 0.9},
            },
            "market": {"trend": "surging"},
        }

        result = module.process(state)

        assert result.get("urgency", 0) > 0.5

    def test_stress_triggers_fight_or_flight(self):
        """Test that high stress triggers fight-or-flight response."""
        module = NeurobiologyModule()

        state = {
            "agent": {
                "neuro_state": {"stress_level": 0.85},
            },
            "market": {"trend": "crashing", "volatility": 0.8},
        }

        result = module.process(state)

        assert result.get("fight_or_flight", False) is True

    def test_reward_sensitivity_affects_response(self):
        """Test that reward sensitivity affects response magnitude."""
        module = NeurobiologyModule()

        high_sensitivity_state = {
            "agent": {
                "neuro_state": {"reward_sensitivity": 0.9},
            },
            "market": {"price_change_pct": 30.0, "trend": "rising"},
        }

        low_sensitivity_state = {
            "agent": {
                "neuro_state": {"reward_sensitivity": 0.2},
            },
            "market": {"price_change_pct": 30.0, "trend": "rising"},
        }

        result_high = module.process(high_sensitivity_state)
        result_low = module.process(low_sensitivity_state)

        assert result_high["fomo_level"] > result_low["fomo_level"]

    def test_calculate_fomo_from_price_momentum(self):
        """Test FOMO calculation from price momentum."""
        module = NeurobiologyModule()

        fomo = module.calculate_fomo(
            price_change_pct=40.0,
            trend="surging",
            social_buzz=0.8,
        )

        assert fomo > 0.5

    def test_no_fomo_during_decline(self):
        """Test minimal FOMO during price decline."""
        module = NeurobiologyModule()

        fomo = module.calculate_fomo(
            price_change_pct=-20.0,
            trend="falling",
            social_buzz=0.3,
        )

        assert fomo < 0.3

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = NeurobiologyModule()
        module._current_state = NeurobiologicalState(
            fomo_level=0.8,
            stress_level=0.3,
        )

        summary = module.get_state_summary()

        assert "FOMO" in summary or "fomo" in summary.lower()

    def test_reset(self):
        """Test resetting module state."""
        module = NeurobiologyModule()
        module._current_state = NeurobiologicalState(
            fomo_level=0.9,
            stress_level=0.8,
        )

        module.reset()

        assert module._current_state.fomo_level == 0.0
        assert module._current_state.stress_level == 0.0
