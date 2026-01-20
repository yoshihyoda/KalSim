"""Layer 7: Market Structure Module.

Models market microstructure, liquidity, order flow, and short squeeze dynamics.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class MarketStructureModule:
    """Models market structure and its impact on agent decisions.
    
    Tracks liquidity, order flow imbalance, volatility, and short squeeze
    conditions that influence collective behavior.
    
    Attributes:
        name: Module identifier.
        liquidity: Current market liquidity (0.0-1.0).
        price_history: Historical price data for volatility calculation.
    """

    def __init__(self) -> None:
        """Initialize the MarketStructureModule."""
        self.name = "market_structure"
        self.liquidity = 1.0
        self.price_history: list[float] = []
        self._short_interest = 0.0
        self._last_volume = 0

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process market state and agent actions.
        
        Calculates market impact from collective agent behavior.
        
        Args:
            state: Combined state including market data and agent actions.
            
        Returns:
            Dictionary with market structure outputs.
        """
        market = state.get("market", {})
        actions = state.get("agent_actions", [])

        current_price = market.get("price", 0.0)
        volume = market.get("volume", 0)
        short_interest = market.get("short_interest", 0.0)

        if current_price > 0:
            self.price_history.append(current_price)
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]

        self._short_interest = short_interest
        self._last_volume = volume

        buy_count = sum(1 for a in actions if a.get("action") == "BUY")
        sell_count = sum(1 for a in actions if a.get("action") == "SELL")
        total_orders = buy_count + sell_count

        order_flow_imbalance = 0.0
        if total_orders > 0:
            order_flow_imbalance = (buy_count - sell_count) / total_orders

        price_pressure = order_flow_imbalance * 0.1 * (1.0 / max(self.liquidity, 0.1))

        self.liquidity = self._calculate_liquidity(volume, total_orders)

        market_impact = abs(price_pressure) * (2.0 - self.liquidity)

        short_squeeze_pressure = self._calculate_short_squeeze_pressure(
            short_interest, buy_count, self.liquidity
        )

        volatility = self.calculate_volatility()

        return {
            "price_pressure": price_pressure,
            "order_flow_imbalance": order_flow_imbalance,
            "liquidity": self.liquidity,
            "market_impact": market_impact,
            "short_squeeze_pressure": short_squeeze_pressure,
            "volatility": volatility,
            "buy_pressure": buy_count / max(total_orders, 1),
            "sell_pressure": sell_count / max(total_orders, 1),
        }

    def _calculate_liquidity(self, volume: int, order_count: int) -> float:
        """Calculate market liquidity.
        
        Higher volume and more orders reduce liquidity.
        
        Args:
            volume: Trading volume.
            order_count: Number of orders from agents.
            
        Returns:
            Liquidity value between 0.1 and 1.0.
        """
        base_volume = 10000000
        volume_factor = base_volume / max(volume, base_volume)

        order_factor = 1.0 - (order_count * 0.01)
        order_factor = max(0.5, order_factor)

        liquidity = self.liquidity * 0.9 + (volume_factor * order_factor) * 0.1
        return max(0.1, min(1.0, liquidity))

    def _calculate_short_squeeze_pressure(
        self,
        short_interest: float,
        buy_count: int,
        liquidity: float,
    ) -> float:
        """Calculate short squeeze pressure.
        
        High short interest + buying pressure + low liquidity = squeeze.
        
        Args:
            short_interest: Short interest percentage.
            buy_count: Number of buy orders.
            liquidity: Current liquidity.
            
        Returns:
            Short squeeze pressure score.
        """
        if short_interest < 20:
            return 0.0

        si_factor = min(short_interest / 100, 1.5)
        buy_factor = min(buy_count / 20, 1.0)
        liquidity_factor = 1.0 - liquidity

        pressure = si_factor * buy_factor * (1 + liquidity_factor)
        return min(pressure, 2.0)

    def calculate_volatility(self, window: int = 10) -> float:
        """Calculate price volatility from recent history.
        
        Uses standard deviation of returns.
        
        Args:
            window: Number of periods to consider.
            
        Returns:
            Volatility measure.
        """
        if len(self.price_history) < 3:
            return 0.0

        recent = self.price_history[-window:]
        if len(recent) < 2:
            return 0.0

        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                ret = (recent[i] - recent[i - 1]) / recent[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def get_state_summary(self) -> str:
        """Get a summary of current market structure state.
        
        Returns:
            Human-readable state summary.
        """
        volatility = self.calculate_volatility()
        return (
            f"Liquidity: {self.liquidity:.2f}, "
            f"Volatility: {volatility:.4f}, "
            f"Short Interest: {self._short_interest:.1f}%"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self.liquidity = 1.0
        self.price_history = []
        self._short_interest = 0.0
        self._last_volume = 0
