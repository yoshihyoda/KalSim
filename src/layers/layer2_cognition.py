"""Layer 2: Cognition Module.

Models cognitive biases and their effects on information processing.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CognitiveBiases:
    """Collection of cognitive bias strengths.
    
    Attributes:
        social_proof: Tendency to follow the crowd (0.0-1.0).
        confirmation_bias: Tendency to favor confirming information (0.0-1.0).
        anchoring: Tendency to anchor on initial values (0.0-1.0).
        loss_aversion: Tendency to weight losses more than gains (0.0-1.0).
        bandwagon: Tendency to join trends (0.0-1.0).
        overconfidence: Tendency to overestimate own knowledge (0.0-1.0).
    """
    social_proof: float = 0.5
    confirmation_bias: float = 0.5
    anchoring: float = 0.5
    loss_aversion: float = 0.6
    bandwagon: float = 0.5
    overconfidence: float = 0.5


class CognitionModule:
    """Models cognitive biases in information processing.
    
    Handles social proof, confirmation bias, anchoring, and loss aversion.
    
    Attributes:
        name: Module identifier.
    """

    def __init__(self) -> None:
        """Initialize the CognitionModule."""
        self.name = "cognition"
        self._current_biases = CognitiveBiases()
        self._active_biases: list[str] = []

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process cognitive state and apply biases.
        
        Args:
            state: Combined state including agent, social, and market data.
            
        Returns:
            Dictionary with cognition outputs.
        """
        agent = state.get("agent", {})
        social = state.get("social", {})
        market = state.get("market", {})
        information = state.get("information", {})

        self._active_biases = []
        outputs: dict[str, Any] = {}

        adjusted_belief = self._apply_social_proof(agent, social, outputs)
        outputs["adjusted_belief"] = adjusted_belief

        info_acceptance = self._apply_confirmation_bias(agent, information, outputs)
        outputs["information_acceptance"] = info_acceptance

        self._apply_anchoring(agent, market, outputs)

        self._apply_loss_aversion(agent, market, outputs)

        self._apply_bandwagon(agent, social, outputs)

        outputs["active_biases"] = self._active_biases.copy()
        outputs["cognitive_load"] = len(self._active_biases) / 5

        return outputs

    def _apply_social_proof(
        self,
        agent: dict[str, Any],
        social: dict[str, Any],
        outputs: dict[str, Any],
    ) -> float:
        """Apply social proof bias.
        
        Adjusts beliefs toward consensus based on peer count and strength.
        
        Args:
            agent: Agent data.
            social: Social context data.
            outputs: Output dictionary to update.
            
        Returns:
            Adjusted belief value.
        """
        beliefs = agent.get("beliefs", {})
        current_belief = beliefs.get("gme_bullish", 0.5)

        consensus_view = social.get("consensus_view", "neutral")
        consensus_strength = social.get("consensus_strength", 0.0)
        peer_count = social.get("peer_count", 0)

        if consensus_view == "bullish":
            consensus_value = consensus_strength
        elif consensus_view == "bearish":
            consensus_value = -consensus_strength
        else:
            consensus_value = 0.0

        peer_factor = min(peer_count / 50, 1.0)
        social_proof_effect = (
            self._current_biases.social_proof
            * consensus_strength
            * peer_factor
        )

        if social_proof_effect > 0.2:
            self._active_biases.append("social_proof")

        adjustment = consensus_value * social_proof_effect * 0.3
        adjusted_belief = current_belief + adjustment

        return max(0.0, min(1.0, adjusted_belief))

    def _apply_confirmation_bias(
        self,
        agent: dict[str, Any],
        information: dict[str, Any],
        outputs: dict[str, Any],
    ) -> float:
        """Apply confirmation bias.
        
        Filters information that contradicts existing beliefs.
        
        Args:
            agent: Agent data.
            information: Information context.
            outputs: Output dictionary to update.
            
        Returns:
            Information acceptance level.
        """
        contrary = information.get("contrary_to_belief", False)

        if not contrary:
            return 1.0

        bias_strength = self._current_biases.confirmation_bias
        acceptance = 1.0 - (bias_strength * 0.7)

        if bias_strength > 0.3:
            self._active_biases.append("confirmation_bias")

        return acceptance

    def _apply_anchoring(
        self,
        agent: dict[str, Any],
        market: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Apply anchoring bias.
        
        Anchors perception to initial or reference prices.
        
        Args:
            agent: Agent data.
            market: Market data.
            outputs: Output dictionary to update.
        """
        anchor_price = agent.get("anchor_price", 0)
        current_price = market.get("current_price", anchor_price)

        if anchor_price <= 0:
            outputs["anchoring_effect"] = 0
            outputs["perceived_overvaluation"] = 0
            return

        price_deviation = (current_price - anchor_price) / anchor_price
        anchoring_effect = self._current_biases.anchoring * abs(price_deviation)

        if price_deviation > 1.0:
            perceived_overvaluation = anchoring_effect * (price_deviation / 10)
        else:
            perceived_overvaluation = 0

        if anchoring_effect > 0.2:
            self._active_biases.append("anchoring")

        outputs["anchoring_effect"] = anchoring_effect
        outputs["perceived_overvaluation"] = perceived_overvaluation

    def _apply_loss_aversion(
        self,
        agent: dict[str, Any],
        market: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Apply loss aversion bias.
        
        Makes agents reluctant to realize losses.
        
        Args:
            agent: Agent data.
            market: Market data.
            outputs: Output dictionary to update.
        """
        entry_price = agent.get("entry_price", 0)
        current_price = market.get("current_price", entry_price)

        if entry_price <= 0:
            outputs["sell_reluctance"] = 0
            return

        pnl_ratio = (current_price - entry_price) / entry_price

        if pnl_ratio < 0:
            sell_reluctance = self._current_biases.loss_aversion * abs(pnl_ratio)
            self._active_biases.append("loss_aversion")
        else:
            sell_reluctance = 0

        outputs["sell_reluctance"] = min(sell_reluctance, 1.0)

    def _apply_bandwagon(
        self,
        agent: dict[str, Any],
        social: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Apply bandwagon effect.
        
        Increases interest when trend adoption is rapid.
        
        Args:
            agent: Agent data.
            social: Social context.
            outputs: Output dictionary to update.
        """
        adoption_rate = social.get("adoption_rate", 0)
        trend_velocity = social.get("trend_velocity", 0)

        if adoption_rate > 0.5 or trend_velocity > 0.5:
            bandwagon_effect = (
                self._current_biases.bandwagon
                * (adoption_rate + trend_velocity) / 2
            )
            self._active_biases.append("bandwagon")
        else:
            bandwagon_effect = 0

        outputs["bandwagon_effect"] = bandwagon_effect

    def calculate_bias_strength(self, biases: CognitiveBiases) -> float:
        """Calculate overall cognitive bias strength.
        
        Args:
            biases: CognitiveBiases instance.
            
        Returns:
            Overall strength (0.0-1.0).
        """
        values = [
            biases.social_proof,
            biases.confirmation_bias,
            biases.anchoring,
            biases.loss_aversion,
            biases.bandwagon,
            biases.overconfidence,
        ]
        return sum(values) / len(values)

    def get_state_summary(self) -> str:
        """Get a summary of current cognition state.
        
        Returns:
            Human-readable state summary.
        """
        active = ", ".join(self._active_biases) if self._active_biases else "none"
        return (
            f"Active biases: {active}, "
            f"Social proof: {self._current_biases.social_proof:.2f}, "
            f"Confirmation: {self._current_biases.confirmation_bias:.2f}"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self._current_biases = CognitiveBiases()
        self._active_biases = []
