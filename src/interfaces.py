"""Abstract interfaces for dependency injection.

Defines ABCs for all external services to enable testability
and loose coupling between components.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMInterfaceABC(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """Generate text completion from the LLM.

        Args:
            prompt: The input prompt to send to the model.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            stream: Whether to use streaming response.

        Returns:
            Generated text response from the model.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if service is responsive, False otherwise.
        """
        pass


class MarketDataProviderABC(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def get_trending_events(
        self,
        limit: int = 20,
        status: str | None = "open",
    ) -> list[dict[str, Any]]:
        """Fetch trending events/markets.

        Args:
            limit: Maximum number of events to return.
            status: Optional status filter.

        Returns:
            List of event dictionaries.
        """
        pass

    @abstractmethod
    def analyze_trends(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze events to extract trends.

        Args:
            events: List of event data objects.

        Returns:
            Dictionary containing topics and summary.
        """
        pass


class UserPoolProviderABC(ABC):
    """Abstract interface for user persona providers."""

    @abstractmethod
    def fetch_user_pool(self, count: int = 100) -> list[dict[str, Any]]:
        """Fetch user personas.

        Args:
            count: Number of users to fetch.

        Returns:
            List of persona dictionaries.
        """
        pass
