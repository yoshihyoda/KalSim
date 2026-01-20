"""Social Environment module for market and social state management.

Provides centralized state management for market conditions and social dynamics.
Optionally integrates with Kalshi API for real prediction market data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..kalshi import KalshiClient

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Container for market state at a point in time.
    
    Attributes:
        price: Current stock price.
        volume: Trading volume.
        short_interest: Short interest percentage.
        liquidity: Market liquidity (0.0-1.0, 1.0 = highly liquid).
        trend: Overall market trend description.
        timestamp: Time of this state.
        previous_price: Price from previous update for change calculation.
    """
    price: float
    volume: int
    short_interest: float
    liquidity: float
    trend: str
    timestamp: datetime
    previous_price: float | None = None

    @property
    def price_change_pct(self) -> float:
        """Calculate percentage change from previous price."""
        if self.previous_price is None or self.previous_price == 0:
            return 0.0
        return ((self.price - self.previous_price) / self.previous_price) * 100


class SocialEnvironment:
    """Manages the social and market environment state.
    
    Centralizes all environment state including market conditions,
    and provides methods for state updates and queries.
    
    Attributes:
        current_state: Current MarketState.
        price_history: List of historical prices.
    """

    DEFAULT_PRICE: float = 20.0
    DEFAULT_VOLUME: int = 10000000
    DEFAULT_SHORT_INTEREST: float = 140.0
    START_DATE: datetime = datetime(2021, 1, 11, 9, 0)

    def __init__(
        self,
        initial_state: dict[str, Any] | None = None,
        use_kalshi: bool = False,
    ) -> None:
        """Initialize the SocialEnvironment.
        
        Args:
            initial_state: Optional dictionary with initial market state.
            use_kalshi: Whether to enable Kalshi API integration for real data.
        """
        self._kalshi_client: "KalshiClient | None" = None
        self._kalshi_trends: dict[str, Any] | None = None
        self._use_kalshi = use_kalshi
        
        if use_kalshi:
            try:
                from ..kalshi import KalshiClient
                self._kalshi_client = KalshiClient()
                logger.info("Kalshi client initialized for real market data")
            except ImportError as e:
                logger.warning(f"Failed to import KalshiClient: {e}")
                self._use_kalshi = False
        
        if initial_state:
            self.current_state = MarketState(
                price=initial_state.get("price", self.DEFAULT_PRICE),
                volume=initial_state.get("volume", self.DEFAULT_VOLUME),
                short_interest=initial_state.get("short_interest", self.DEFAULT_SHORT_INTEREST),
                liquidity=initial_state.get("liquidity", 1.0),
                trend=initial_state.get("trend", "stable"),
                timestamp=initial_state.get("timestamp", self.START_DATE),
            )
        else:
            self.current_state = MarketState(
                price=self.DEFAULT_PRICE,
                volume=self.DEFAULT_VOLUME,
                short_interest=self.DEFAULT_SHORT_INTEREST,
                liquidity=1.0,
                trend="stable",
                timestamp=self.START_DATE,
            )
        
        self.price_history: list[float] = [self.current_state.price]
        self._community_sentiment: float = 0.0

    def update_state(
        self,
        price: float | None = None,
        volume: int | None = None,
        short_interest: float | None = None,
        liquidity: float | None = None,
        trend: str | None = None,
    ) -> None:
        """Update market state with new values.
        
        Args:
            price: New price (optional).
            volume: New volume (optional).
            short_interest: New short interest (optional).
            liquidity: New liquidity (optional).
            trend: New trend description (optional).
        """
        previous_price = self.current_state.price

        new_price = price if price is not None else self.current_state.price
        new_volume = volume if volume is not None else self.current_state.volume
        new_short_interest = short_interest if short_interest is not None else self.current_state.short_interest
        new_trend = trend if trend is not None else self.current_state.trend

        if liquidity is not None:
            new_liquidity = liquidity
        elif volume is not None:
            new_liquidity = self._calculate_liquidity(new_volume)
        else:
            new_liquidity = self.current_state.liquidity

        self.current_state = MarketState(
            price=new_price,
            volume=new_volume,
            short_interest=new_short_interest,
            liquidity=new_liquidity,
            trend=new_trend,
            timestamp=self.current_state.timestamp,
            previous_price=previous_price,
        )

        self.price_history.append(new_price)

    def _calculate_liquidity(self, volume: int) -> float:
        """Calculate liquidity based on volume.
        
        Higher volume reduces liquidity (more market stress).
        
        Args:
            volume: Trading volume.
            
        Returns:
            Liquidity value between 0.0 and 1.0.
        """
        base_volume = 10000000
        if volume <= base_volume:
            return 1.0
        ratio = base_volume / volume
        return max(0.1, min(1.0, ratio))

    def get_price_trend(self, window: int = 5) -> str:
        """Determine price trend from recent history.
        
        Args:
            window: Number of recent prices to consider.
            
        Returns:
            Trend description: 'surging', 'rising', 'stable', 'falling', 'crashing'.
        """
        if len(self.price_history) < 2:
            return "stable"

        recent = self.price_history[-window:]
        if len(recent) < 2:
            return "stable"

        first = recent[0]
        last = recent[-1]
        
        if first == 0:
            return "stable"
            
        change_pct = ((last - first) / first) * 100

        if change_pct > 20:
            return "surging"
        elif change_pct > 5:
            return "rising"
        elif change_pct < -20:
            return "crashing"
        elif change_pct < -5:
            return "falling"
        else:
            return "stable"

    def advance_time(self, hours: int = 1) -> None:
        """Advance the environment time.
        
        Args:
            hours: Number of hours to advance.
        """
        new_timestamp = self.current_state.timestamp + timedelta(hours=hours)
        self.current_state = MarketState(
            price=self.current_state.price,
            volume=self.current_state.volume,
            short_interest=self.current_state.short_interest,
            liquidity=self.current_state.liquidity,
            trend=self.current_state.trend,
            timestamp=new_timestamp,
            previous_price=self.current_state.previous_price,
        )

    def get_market_info(self) -> dict[str, Any]:
        """Get current market state as dictionary.
        
        Returns:
            Dictionary with current market information.
        """
        return {
            "price": self.current_state.price,
            "volume": self.current_state.volume,
            "short_interest": self.current_state.short_interest,
            "liquidity": self.current_state.liquidity,
            "trend": self.current_state.trend,
            "timestamp": self.current_state.timestamp,
            "price_change_pct": self.current_state.price_change_pct,
        }

    @property
    def community_sentiment(self) -> float:
        """Get current community sentiment."""
        return self._community_sentiment

    @community_sentiment.setter
    def community_sentiment(self, value: float) -> None:
        """Set community sentiment, clamped to [-1, 1]."""
        self._community_sentiment = max(-1.0, min(1.0, value))

    def load_kalshi_trends(self, limit: int = 10) -> dict[str, Any]:
        """Fetch and store trending Kalshi markets.
        
        Args:
            limit: Maximum number of events to fetch.
            
        Returns:
            Dictionary containing topics, tickers, and summary.
        """
        if not self._kalshi_client:
            logger.debug("Kalshi client not initialized, returning empty trends")
            return {}
        
        try:
            events = self._kalshi_client.get_trending_events(limit=limit)
            self._kalshi_trends = self._kalshi_client.analyze_trends(events)
            logger.info(
                f"Loaded {len(self._kalshi_trends.get('topics', []))} trending topics from Kalshi"
            )
            return self._kalshi_trends
        except Exception as e:
            logger.error(f"Failed to load Kalshi trends: {e}")
            return {}

    def get_trending_topics(self) -> list[str]:
        """Get trending topics from Kalshi data.
        
        Returns:
            List of trending topic strings, or empty list if not available.
        """
        if self._kalshi_trends:
            return self._kalshi_trends.get("topics", [])
        return []

    def get_kalshi_summary(self) -> str:
        """Get summary of Kalshi market trends.
        
        Returns:
            Summary string for LLM context.
        """
        if self._kalshi_trends:
            return self._kalshi_trends.get("summary", "")
        return ""

    @property
    def has_kalshi_data(self) -> bool:
        """Check if Kalshi data is loaded."""
        return self._kalshi_trends is not None and len(self.get_trending_topics()) > 0
