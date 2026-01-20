"""Kalshi API Client and Trend Analyzer.

This module handles fetching market data from Kalshi's public API and analyzing it
to identify trending topics for the simulation.
"""

import logging
import time
import requests
from typing import Any, Final

from .interfaces import MarketDataProviderABC

logger = logging.getLogger(__name__)

# Kalshi public API endpoints (public market data + exchange status)
KALSHI_API_URL: Final[str] = "https://api.elections.kalshi.com/trade-api/v2/markets"
KALSHI_EVENTS_API_URL: Final[str] = "https://api.elections.kalshi.com/trade-api/v2/events"
KALSHI_EXCHANGE_STATUS_URL: Final[str] = (
    "https://api.elections.kalshi.com/trade-api/v2/exchange/status"
)


class KalshiClient(MarketDataProviderABC):
    """Client for interacting with Kalshi public API."""
    
    def __init__(self):
        self.session = requests.Session()
        self._market_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
        self._cache_expiry_seconds: int = 30

    def get_exchange_status(self) -> dict[str, Any] | None:
        """Fetch exchange status (maintenance/trading availability)."""
        try:
            response = self.session.get(KALSHI_EXCHANGE_STATUS_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                logger.warning("Unexpected exchange status response type: %s", type(data))
                return None
            return data
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Failed to fetch exchange status: {e}")
            return None

    def get_public_markets(
        self,
        limit: int = 20,
        status: str | None = "open",
        check_exchange_status: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch open markets from Kalshi with caching.
        
        Args:
            limit: Maximum number of markets to return.
            status: Optional status filter (e.g., "open"). Use None for all statuses.
            check_exchange_status: Whether to preflight exchange status before fetching.
            
        Returns:
            List of market dictionaries.
        """
        cache_key = f"public_markets_{limit}_{status}"
        now = time.time()
        
        if cache_key in self._market_cache:
            cached_data, timestamp = self._market_cache[cache_key]
            if (now - timestamp) < self._cache_expiry_seconds:
                logger.debug("Returning cached market data for key: %s", cache_key)
                return cached_data
        
        try:
            if check_exchange_status:
                exchange_status = self.get_exchange_status()
                if exchange_status:
                    exchange_active = exchange_status.get("exchange_active")
                    trading_active = exchange_status.get("trading_active")
                    if exchange_active is False:
                        resume_time = exchange_status.get("exchange_estimated_resume_time")
                        logger.warning(
                            "Exchange inactive; skipping markets. resume_time=%s",
                            resume_time,
                        )
                        return []
                    if trading_active is False:
                        logger.info(
                            "Trading inactive; market data may be stale or limited."
                        )

            params: dict[str, Any] = {
                "limit": 100  # Fetch more to filter
            }
            if status:
                normalized = status.strip().lower()
                if normalized == "active":
                    normalized = "open"
                allowed_statuses = {"unopened", "open", "closed", "settled"}
                if normalized not in allowed_statuses:
                    logger.warning("Invalid status filter '%s'; ignoring.", normalized)
                    normalized = None
                if normalized:
                    params["status"] = normalized
            response = self.session.get(KALSHI_API_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            markets = data.get("markets", [])
            
            markets_sorted = sorted(
                markets, 
                key=lambda m: (
                    m.get("volume_24h")
                    or m.get("volume")
                    or m.get("open_interest")
                    or 0
                ),
                reverse=True
            )
            
            result = markets_sorted[:limit]
            self._market_cache[cache_key] = (result, now)
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Kalshi markets: {e}")
            return []

    def get_trending_events(
        self,
        limit: int = 20,
        status: str | None = "open",
        check_exchange_status: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch trending events from Kalshi (approx by recent activity)."""
        try:
            if check_exchange_status:
                exchange_status = self.get_exchange_status()
                if exchange_status:
                    exchange_active = exchange_status.get("exchange_active")
                    trading_active = exchange_status.get("trading_active")
                    if exchange_active is False:
                        resume_time = exchange_status.get("exchange_estimated_resume_time")
                        logger.warning(
                            "Exchange inactive; skipping events. resume_time=%s",
                            resume_time,
                        )
                        return []
                    if trading_active is False:
                        logger.info(
                            "Trading inactive; event data may be stale or limited."
                        )

            params: dict[str, Any] = {
                "limit": max(limit * 5, 100),
                "with_nested_markets": True,
            }
            if status:
                normalized = status.strip().lower()
                if normalized == "active":
                    normalized = "open"
                allowed_statuses = {"unopened", "open", "closed", "settled"}
                if normalized not in allowed_statuses:
                    logger.warning("Invalid status filter '%s'; ignoring.", normalized)
                    normalized = None
                if normalized:
                    params["status"] = normalized

            response = self.session.get(KALSHI_EVENTS_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            events = data.get("events", [])

            def event_score(event: dict[str, Any]) -> tuple[int, int, int]:
                markets = event.get("markets", []) or []
                volume_24h = sum(int(m.get("volume_24h") or 0) for m in markets)
                if volume_24h == 0:
                    volume_24h = sum(int(m.get("volume") or 0) for m in markets)
                liquidity = sum(int(m.get("liquidity") or 0) for m in markets)
                open_interest = sum(int(m.get("open_interest") or 0) for m in markets)
                return (volume_24h, liquidity, open_interest)

            events_sorted = sorted(events, key=event_score, reverse=True)
            return events_sorted[:limit]
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Kalshi events: {e}")
            return []

    def get_event_details(self, event_ticker: str) -> dict[str, Any] | None:
        """Fetch event details with nested markets."""
        if not event_ticker:
            return None
        try:
            response = self.session.get(
                f"{KALSHI_EVENTS_API_URL}/{event_ticker}",
                params={"with_nested_markets": True},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            event = data.get("event")
            if not isinstance(event, dict):
                return None
            return event
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Kalshi event details: {e}")
            return None

    def summarize_event(self, event: dict[str, Any]) -> str:
        """Build a compact summary string for persona generation."""
        title = event.get("title") or "Unknown event"
        category = event.get("category") or "Uncategorized"
        sub_title = event.get("sub_title") or ""
        event_ticker = event.get("event_ticker") or ""

        summary_lines = [f"Event: {title} ({category})"]
        if sub_title:
            summary_lines.append(f"Subtitle: {sub_title}")
        if event_ticker:
            summary_lines.append(f"Event ticker: {event_ticker}")

        markets = event.get("markets", []) or []
        if markets:
            def market_score(m: dict[str, Any]) -> tuple[int, int, int]:
                volume_24h = int(m.get("volume_24h") or 0)
                volume = int(m.get("volume") or 0)
                open_interest = int(m.get("open_interest") or 0)
                return (volume_24h, volume, open_interest)

            top_markets = sorted(markets, key=market_score, reverse=True)[:5]
            market_titles = []
            for market in top_markets:
                market_title = self._clean_title(market.get("title", ""))
                market_subtitle = market.get("subtitle") or market.get("yes_sub_title") or ""
                if market_subtitle:
                    market_titles.append(f"{market_title} ({market_subtitle})")
                else:
                    market_titles.append(market_title)
            summary_lines.append("Top markets: " + "; ".join(market_titles))

        return " | ".join(summary_lines)
    def _clean_title(self, title: str) -> str:
        """Clean market title for display."""
        if not title:
            return "Unknown Market"
            
        # Remove repetitive "yes "/"no " prefixes common in Kalshi markets
        # Case insensitive replacement
        import re
        cleaned = re.sub(r'\b(yes|no)\s+', '', title, flags=re.IGNORECASE)
        
        # Truncate if very long
        if len(cleaned) > 80:
            cleaned = cleaned[:77] + "..."
            
        return cleaned.strip()

    def analyze_trends(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze events to extract trends and keywords.
        
        Args:
            events: List of event data objects.
            
        Returns:
            Dictionary containing 'topics' (list of strings) and 'summary' (str).
        """
        # Extract and clean data, ensuring alignment
        trending_data = []
        for event in events:
            title = event.get("title", "")
            slug = self._get_event_slug(event)
            event_ticker = event.get("event_ticker")
            series_ticker = event.get("series_ticker")
            
            if title and slug:
                cleaned = self._clean_title(title)
                if cleaned:
                    trending_data.append(
                        {
                            "title": cleaned,
                            "slug": slug,
                            "event_ticker": event_ticker,
                            "series_ticker": series_ticker,
                        }
                    )
        
        # Take top 8
        top_items = trending_data[:8]
        if not top_items:
            return {
                "topics": ["General Market"],
                "tickers": [""],
                "summary": "No market data available.",
            }

        topics = [item["title"] for item in top_items]
        tickers = [item["slug"] for item in top_items]
        event_tickers = [item.get("event_ticker") or "" for item in top_items]
        series_tickers = [item.get("series_ticker") or "" for item in top_items]
        
        # Summary for LLM
        summary = f"Top trending Kalshi markets: {', '.join(topics)}"
        
        return {
            "topics": topics,
            "tickers": tickers,
            "event_tickers": event_tickers,
            "series_tickers": series_tickers,
            "summary": summary,
        }

    def _get_event_slug(self, event: dict[str, Any]) -> str | None:
        """Derive the URL slug for an event (prefer series page for web)."""
        # Kalshi web UI typically resolves series tickers more reliably than event tickers.
        series_ticker = event.get("series_ticker")
        if series_ticker:
            return series_ticker
        event_ticker = event.get("event_ticker")
        if event_ticker:
            return event_ticker
        return None

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    client = KalshiClient()
    events = client.get_trending_events()
    analysis = client.analyze_trends(events)
    print(f"Analysis: {analysis}")
