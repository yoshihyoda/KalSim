"""Layer 1: Neurobiology Module.

Models neurobiological responses including FOMO, dopamine, and stress.
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NeurobiologicalState:
    """State of neurobiological processes.
    
    Attributes:
        fomo_level: Fear of missing out intensity (0.0-1.0).
        dopamine_response: Dopamine/reward response level (0.0-1.0).
        stress_level: Cortisol/stress level (0.0-1.0).
        reward_sensitivity: Sensitivity to rewards (0.0-1.0).
        habituation: Habituation to stimuli (0.0-1.0).
    """
    fomo_level: float = 0.0
    dopamine_response: float = 0.5
    stress_level: float = 0.0
    reward_sensitivity: float = 0.5
    habituation: float = 0.0


class NeurobiologyModule:
    """Models neurobiological responses to market stimuli.
    
    Handles FOMO, dopamine response, stress, and fight-or-flight.
    
    Attributes:
        name: Module identifier.
    """

    FOMO_THRESHOLD: float = 0.7
    STRESS_THRESHOLD: float = 0.8

    def __init__(self) -> None:
        """Initialize the NeurobiologyModule."""
        self.name = "neurobiology"
        self._current_state = NeurobiologicalState()

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process neurobiological response to stimuli.
        
        Args:
            state: Combined state including agent and market data.
            
        Returns:
            Dictionary with neurobiology outputs.
        """
        agent = state.get("agent", {})
        market = state.get("market", {})
        social = state.get("social", {})

        neuro_state = agent.get("neuro_state", {})
        current_fomo = neuro_state.get("fomo_level", 0.0)
        current_stress = neuro_state.get("stress_level", 0.0)
        current_dopamine = neuro_state.get("dopamine_response", 0.5)
        reward_sensitivity = neuro_state.get("reward_sensitivity", 0.5)
        habituation = neuro_state.get("habituation", 0.0)

        price_change = market.get("price_change_pct", 0.0)
        trend = market.get("trend", "stable")
        volatility = market.get("volatility", 0.0)

        social_buzz = social.get("sentiment", 0.0)

        new_fomo = self._update_fomo(
            current_fomo, price_change, trend, social_buzz, reward_sensitivity
        )

        new_stress = self._update_stress(
            current_stress, price_change, trend, volatility
        )

        portfolio = agent.get("portfolio", {})
        unrealized_pnl = portfolio.get("unrealized_pnl_pct", 0.0)

        new_dopamine, new_habituation = self._update_dopamine(
            current_dopamine, unrealized_pnl, price_change, habituation
        )

        urgency = new_fomo * 0.7 + (1 - self._current_state.habituation) * 0.3
        fight_or_flight = new_stress >= self.STRESS_THRESHOLD

        self._current_state = NeurobiologicalState(
            fomo_level=new_fomo,
            dopamine_response=new_dopamine,
            stress_level=new_stress,
            reward_sensitivity=reward_sensitivity,
            habituation=new_habituation,
        )

        return {
            "fomo_level": new_fomo,
            "dopamine_response": new_dopamine,
            "stress_level": new_stress,
            "habituation": new_habituation,
            "urgency": urgency,
            "fight_or_flight": fight_or_flight,
            "reward_sensitivity": reward_sensitivity,
        }

    def _update_fomo(
        self,
        current: float,
        price_change: float,
        trend: str,
        social_buzz: float,
        sensitivity: float,
    ) -> float:
        """Update FOMO level based on market and social signals.
        
        Args:
            current: Current FOMO level.
            price_change: Price change percentage.
            trend: Market trend.
            social_buzz: Social sentiment/activity.
            sensitivity: Reward sensitivity.
            
        Returns:
            Updated FOMO level.
        """
        fomo_delta = 0.0

        if price_change > 0:
            fomo_delta += (price_change / 100) * 0.5 * (1 + sensitivity)

        if trend in ["surging", "rising"]:
            trend_boost = 0.3 if trend == "surging" else 0.15
            fomo_delta += trend_boost

        if social_buzz > 0:
            fomo_delta += social_buzz * 0.2

        fomo_delta -= current * 0.1

        new_fomo = current + fomo_delta
        return max(0.0, min(1.0, new_fomo))

    def _update_stress(
        self,
        current: float,
        price_change: float,
        trend: str,
        volatility: float,
    ) -> float:
        """Update stress level based on market conditions.
        
        Args:
            current: Current stress level.
            price_change: Price change percentage.
            trend: Market trend.
            volatility: Market volatility.
            
        Returns:
            Updated stress level.
        """
        stress_delta = 0.0

        if price_change < 0:
            stress_delta += abs(price_change / 100) * 0.4

        if trend in ["crashing", "falling"]:
            trend_stress = 0.4 if trend == "crashing" else 0.2
            stress_delta += trend_stress

        stress_delta += volatility * 0.3

        stress_delta -= current * 0.15

        new_stress = current + stress_delta
        return max(0.0, min(1.0, new_stress))

    def _update_dopamine(
        self,
        current: float,
        unrealized_pnl: float,
        price_change: float,
        habituation: float,
    ) -> tuple[float, float]:
        """Update dopamine response and habituation.
        
        Args:
            current: Current dopamine level.
            unrealized_pnl: Unrealized P&L percentage.
            price_change: Price change percentage.
            habituation: Current habituation level.
            
        Returns:
            Tuple of (new dopamine, new habituation).
        """
        dopamine_delta = 0.0

        if unrealized_pnl > 0:
            gain_response = (unrealized_pnl / 100) * 0.3
            dopamine_delta += gain_response * (1 - habituation)

        if price_change > 0:
            dopamine_delta += (price_change / 100) * 0.2 * (1 - habituation)

        dopamine_delta -= (current - 0.5) * 0.1

        new_dopamine = current + dopamine_delta
        new_dopamine = max(0.0, min(1.0, new_dopamine))

        if dopamine_delta > 0:
            new_habituation = min(habituation + 0.05, 0.8)
        else:
            new_habituation = max(habituation - 0.02, 0.0)

        return new_dopamine, new_habituation

    def calculate_fomo(
        self,
        price_change_pct: float,
        trend: str,
        social_buzz: float,
    ) -> float:
        """Calculate FOMO level from inputs.
        
        Args:
            price_change_pct: Price change percentage.
            trend: Market trend.
            social_buzz: Social buzz level.
            
        Returns:
            FOMO level (0.0-1.0).
        """
        fomo = 0.0

        if price_change_pct > 0:
            fomo += min(price_change_pct / 50, 0.5)

        if trend == "surging":
            fomo += 0.3
        elif trend == "rising":
            fomo += 0.15

        fomo += social_buzz * 0.2

        return max(0.0, min(1.0, fomo))

    def get_state_summary(self) -> str:
        """Get a summary of current neurobiological state.
        
        Returns:
            Human-readable state summary.
        """
        return (
            f"FOMO: {self._current_state.fomo_level:.2f}, "
            f"Dopamine: {self._current_state.dopamine_response:.2f}, "
            f"Stress: {self._current_state.stress_level:.2f}, "
            f"Habituation: {self._current_state.habituation:.2f}"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self._current_state = NeurobiologicalState()
