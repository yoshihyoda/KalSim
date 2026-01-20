"""Layer 5: Collective Identity Module.

Models in-group/out-group dynamics and identity-based information filtering.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class IdentityGroup(Enum):
    """Identity groups for collective identity modeling."""
    WSB_APE = auto()
    RETAIL_INVESTOR = auto()
    INSTITUTIONAL = auto()
    SKEPTIC = auto()
    NEUTRAL = auto()


@dataclass
class IdentityState:
    """State of an agent's collective identity.
    
    Attributes:
        primary_group: Primary identity group membership.
        group_identification: Strength of identification (0.0-1.0).
        in_group_trust: Trust level for in-group members.
        out_group_trust: Trust level for out-group members.
    """
    primary_group: IdentityGroup
    group_identification: float = 0.5
    in_group_trust: float = 0.7
    out_group_trust: float = 0.3


class IdentityModule:
    """Models collective identity and its effects on trust and behavior.
    
    Handles group assignment, trust adjustment, and conformity pressure.
    
    Attributes:
        name: Module identifier.
    """

    def __init__(self) -> None:
        """Initialize the IdentityModule."""
        self.name = "collective_identity"
        self._current_identity: IdentityState | None = None

    def assign_identity(self, persona: dict[str, Any]) -> IdentityState:
        """Assign an identity group based on persona traits.
        
        Args:
            persona: Agent persona dictionary.
            
        Returns:
            IdentityState with assigned group.
        """
        traits = set(t.lower() for t in persona.get("personality_traits", []))
        interests = set(i.lower() for i in persona.get("interests", []))
        beliefs = persona.get("beliefs", {})

        risk_tolerance = beliefs.get("risk_tolerance", "moderate")
        trust_institutions = beliefs.get("trust_in_institutions", "moderate")

        wsb_indicators = {"wsb", "memes", "reddit", "crypto", "yolo"}
        wsb_match = len(interests & wsb_indicators)

        if wsb_match >= 2 or (wsb_match >= 1 and risk_tolerance == "high"):
            group = IdentityGroup.WSB_APE
            identification = 0.7 + (wsb_match * 0.1)
            in_group_trust = 0.85
            out_group_trust = 0.2
        elif trust_institutions == "high" and "analytical" in traits:
            if "quantitative" in " ".join(interests).lower():
                group = IdentityGroup.INSTITUTIONAL
            else:
                group = IdentityGroup.RETAIL_INVESTOR
            identification = 0.5
            in_group_trust = 0.6
            out_group_trust = 0.4
        elif "skeptical" in traits or trust_institutions == "low":
            group = IdentityGroup.SKEPTIC
            identification = 0.4
            in_group_trust = 0.5
            out_group_trust = 0.3
        else:
            group = IdentityGroup.NEUTRAL
            identification = 0.3
            in_group_trust = 0.5
            out_group_trust = 0.5

        return IdentityState(
            primary_group=group,
            group_identification=min(identification, 1.0),
            in_group_trust=in_group_trust,
            out_group_trust=out_group_trust,
        )

    def adjust_trust(
        self,
        agent_identity: IdentityState,
        source_group: IdentityGroup,
        base_trust: float,
    ) -> float:
        """Adjust trust based on group membership.
        
        In-group sources receive trust boost, out-group reduced.
        
        Args:
            agent_identity: Agent's identity state.
            source_group: Group of the message source.
            base_trust: Base trust level (0.0-1.0).
            
        Returns:
            Adjusted trust level.
        """
        if source_group == agent_identity.primary_group:
            adjustment = agent_identity.in_group_trust
        elif source_group == IdentityGroup.NEUTRAL:
            adjustment = (agent_identity.in_group_trust + agent_identity.out_group_trust) / 2
        else:
            adjustment = agent_identity.out_group_trust

        adjusted = base_trust * (adjustment / 0.5)
        return max(0.0, min(1.0, adjusted))

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process identity state for an agent.
        
        Args:
            state: Combined state including agent persona.
            
        Returns:
            Dictionary with identity outputs.
        """
        agent = state.get("agent", {})
        persona = agent.get("persona", {})
        social = state.get("social", {})

        identity = self.assign_identity(persona)
        self._current_identity = identity

        dominant_group = social.get("dominant_group", "")
        group_alignment = 0.0
        if dominant_group.upper() == "WSB" and identity.primary_group == IdentityGroup.WSB_APE:
            group_alignment = identity.group_identification

        return {
            "identity_group": identity.primary_group.name,
            "group_identification": identity.group_identification,
            "in_group_trust": identity.in_group_trust,
            "out_group_trust": identity.out_group_trust,
            "group_alignment": group_alignment,
            "identity_salience": identity.group_identification * 0.8,
        }

    def calculate_conformity_pressure(
        self,
        agent_identity: IdentityState,
        group_consensus: dict[str, Any],
    ) -> float:
        """Calculate pressure to conform to group consensus.
        
        Args:
            agent_identity: Agent's identity state.
            group_consensus: Current group consensus (action, strength).
            
        Returns:
            Conformity pressure score (0.0-1.0).
        """
        consensus_strength = group_consensus.get("strength", 0.0)
        pressure = agent_identity.group_identification * consensus_strength
        return min(pressure, 1.0)

    def update_salience(
        self,
        identity: IdentityState,
        context: dict[str, Any],
    ) -> IdentityState:
        """Update identity salience based on context.
        
        Group identity becomes more salient when group is mentioned,
        threatened, or when interacting with in-group members.
        
        Args:
            identity: Current identity state.
            context: Context with group mentions, interactions, threats.
            
        Returns:
            Updated IdentityState.
        """
        mentions = context.get("group_mentions", 0)
        interactions = context.get("in_group_interactions", 0)
        threat = context.get("threat_to_group", False)

        salience_boost = 0.0
        salience_boost += min(mentions * 0.02, 0.1)
        salience_boost += min(interactions * 0.03, 0.15)
        if threat:
            salience_boost += 0.2

        new_identification = min(
            identity.group_identification + salience_boost, 1.0
        )

        return IdentityState(
            primary_group=identity.primary_group,
            group_identification=new_identification,
            in_group_trust=identity.in_group_trust,
            out_group_trust=identity.out_group_trust,
        )

    def get_state_summary(self) -> str:
        """Get a summary of current identity state.
        
        Returns:
            Human-readable state summary.
        """
        if self._current_identity is None:
            return "No identity assigned"

        return (
            f"Group: {self._current_identity.primary_group.name}, "
            f"Identification: {self._current_identity.group_identification:.2f}, "
            f"In-group trust: {self._current_identity.in_group_trust:.2f}"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self._current_identity = None
