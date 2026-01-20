"""Tests for Layer 5: Collective Identity module."""

import pytest

from src.layers.layer5_collective_identity import (
    IdentityModule,
    IdentityState,
    IdentityGroup,
)


class TestIdentityGroup:
    """Tests for IdentityGroup enum."""

    def test_identity_groups_exist(self):
        """Test that required identity groups exist."""
        assert IdentityGroup.WSB_APE
        assert IdentityGroup.RETAIL_INVESTOR
        assert IdentityGroup.INSTITUTIONAL
        assert IdentityGroup.SKEPTIC


class TestIdentityState:
    """Tests for IdentityState dataclass."""

    def test_state_creation(self):
        """Test creating IdentityState."""
        state = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.9,
            in_group_trust=0.85,
            out_group_trust=0.2,
        )
        assert state.primary_group == IdentityGroup.WSB_APE
        assert state.group_identification == 0.9


class TestIdentityModule:
    """Tests for IdentityModule class."""

    def test_module_initialization(self):
        """Test initializing IdentityModule."""
        module = IdentityModule()
        assert module.name == "collective_identity"

    def test_assign_identity_high_risk_wsb_interests(self):
        """Test assigning WSB identity to high-risk meme trader."""
        module = IdentityModule()

        persona = {
            "personality_traits": ["risk-seeking", "community-focused"],
            "interests": ["wsb", "memes", "reddit"],
            "beliefs": {"risk_tolerance": "high", "trust_in_institutions": "low"},
        }

        identity = module.assign_identity(persona)

        assert identity.primary_group == IdentityGroup.WSB_APE

    def test_assign_identity_conservative(self):
        """Test assigning identity to conservative persona."""
        module = IdentityModule()

        persona = {
            "personality_traits": ["cautious", "analytical"],
            "interests": ["value investing", "bonds"],
            "beliefs": {"risk_tolerance": "low", "trust_in_institutions": "high"},
        }

        identity = module.assign_identity(persona)

        assert identity.primary_group in [
            IdentityGroup.RETAIL_INVESTOR,
            IdentityGroup.SKEPTIC,
        ]

    def test_in_group_trust_boost(self):
        """Test that in-group messages receive trust boost."""
        module = IdentityModule()

        agent_identity = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.9,
            in_group_trust=0.85,
            out_group_trust=0.2,
        )

        message_source_group = IdentityGroup.WSB_APE
        base_trust = 0.5

        adjusted_trust = module.adjust_trust(
            agent_identity, message_source_group, base_trust
        )

        assert adjusted_trust > base_trust

    def test_out_group_trust_reduction(self):
        """Test that out-group messages receive trust reduction."""
        module = IdentityModule()

        agent_identity = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.9,
            in_group_trust=0.85,
            out_group_trust=0.2,
        )

        message_source_group = IdentityGroup.INSTITUTIONAL
        base_trust = 0.5

        adjusted_trust = module.adjust_trust(
            agent_identity, message_source_group, base_trust
        )

        assert adjusted_trust < base_trust

    def test_process_state(self):
        """Test processing identity state."""
        module = IdentityModule()

        state = {
            "agent": {
                "persona": {
                    "personality_traits": ["impulsive", "social"],
                    "interests": ["wsb", "crypto"],
                    "beliefs": {"risk_tolerance": "high"},
                }
            },
            "social": {"dominant_group": "WSB"},
        }

        result = module.process(state)

        assert "identity_group" in result
        assert "group_identification" in result
        assert "in_group_trust" in result

    def test_conformity_pressure(self):
        """Test calculation of conformity pressure."""
        module = IdentityModule()

        agent_identity = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.9,
        )

        group_consensus = {"action": "HOLD", "strength": 0.8}

        pressure = module.calculate_conformity_pressure(
            agent_identity, group_consensus
        )

        assert pressure > 0

    def test_identity_salience_update(self):
        """Test updating identity salience based on context."""
        module = IdentityModule()

        identity = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.7,
        )

        context = {
            "group_mentions": 10,
            "in_group_interactions": 5,
            "threat_to_group": True,
        }

        updated = module.update_salience(identity, context)

        assert updated.group_identification >= identity.group_identification

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = IdentityModule()
        module._current_identity = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.8,
        )

        summary = module.get_state_summary()

        assert "WSB" in summary or "group" in summary.lower()

    def test_reset(self):
        """Test resetting module state."""
        module = IdentityModule()
        module._current_identity = IdentityState(
            primary_group=IdentityGroup.WSB_APE,
            group_identification=0.9,
        )

        module.reset()

        assert module._current_identity is None
