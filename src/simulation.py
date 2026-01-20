"""Simulation engine module.

Manages the main simulation loop and agent interactions.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .agent import Agent, MarketInfo, SocialMediaInfo, ActionResult
from .config import (
    SIMULATION_DAYS,
    AGENTS_COUNT,
    TIME_STEP_HOURS,
    STEPS_PER_DAY,
    PERSONA_FILE,
    TWEETS_FILE,
    SIMULATION_LOG_FILE,
    ensure_directories,
)
from .llm_interface import LlamaInterface
from .kalshi import KalshiClient
from .interfaces import LLMInterfaceABC, MarketDataProviderABC, UserPoolProviderABC

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Base exception for simulation errors."""
    pass


class Simulation:
    """Main simulation engine for social dynamics modeling.
    
    Orchestrates agent creation, time-step execution, and result collection.
    
    Attributes:
        days: Number of days to simulate.
        agent_count: Number of agents in the simulation.
        agents: List of Agent instances.
        llm: LlamaInterface for agent decisions.
        simulation_log: Collected action results.
    """
    
    BASE_PRICE: float = 20.0
    START_DATE: datetime = datetime(2021, 1, 11, 9, 0)
    
    def __init__(
        self,
        days: int = SIMULATION_DAYS,
        agent_count: int = AGENTS_COUNT,
        persona_file: Path = PERSONA_FILE,
        tweets_file: Path = TWEETS_FILE,
        mock_llm: bool = False,
        custom_agents: list[dict[str, Any]] | None = None,
        use_kalshi: bool = False,
        market_topic: str | None = None,
        llm_provider: LLMInterfaceABC | None = None,
        market_provider: MarketDataProviderABC | None = None,
        user_pool_provider: UserPoolProviderABC | None = None,
    ) -> None:
        """Initialize the simulation.
        
        Args:
            days: Number of simulation days.
            agent_count: Number of agents to create.
            persona_file: Path to persona JSON file (deprecated).
            tweets_file: Path to tweets CSV file (deprecated).
            mock_llm: Whether to run in mock mode.
            custom_agents: Optional personas to use instead of file/defaults.
            use_kalshi: Whether to use real Kalshi market data.
            market_topic: The market topic to simulate discussions about.
            llm_provider: Optional LLM interface (for dependency injection).
            market_provider: Optional market data provider (for dependency injection).
            user_pool_provider: Optional user pool provider (for dependency injection).
        """
        ensure_directories()
        
        self.days = days
        self.agent_count = agent_count
        self.persona_file = persona_file
        self.tweets_file = tweets_file
        self.mock_llm = mock_llm
        self.custom_agents = custom_agents or []
        self.use_kalshi = use_kalshi or not mock_llm
        self.market_topic = market_topic or "prediction markets"
        
        self.agents: list[Agent] = []
        self.llm: LLMInterfaceABC | None = llm_provider
        self.simulation_log: list[dict[str, Any]] = []
        self.seed_tweets: list[str] = []
        self.stop_requested = False
        
        self._current_price = self.BASE_PRICE
        self._current_time = self.START_DATE
        self._current_step_index = 0
        self._price_history: list[float] = [self.BASE_PRICE]
        self._community_sentiment = 0.0
        
        self._kalshi_client: MarketDataProviderABC | None = market_provider
        self._kalshi_analysis: dict[str, Any] | None = None
        self._user_pool_provider: UserPoolProviderABC | None = user_pool_provider
        
        if self.use_kalshi and self._kalshi_client is None:
            self._kalshi_client = KalshiClient()
            logger.info("Kalshi client initialized for real market data")
        
        logger.info(
            f"Simulation initialized: {days} days, {agent_count} agents, "
            f"mock_llm={mock_llm}, use_kalshi={self.use_kalshi}"
        )
    
    def setup(self) -> None:
        """Set up simulation components."""
        logger.info("Setting up simulation...")
        
        if self.llm is None:
            self.llm = LlamaInterface(mock_mode=self.mock_llm)
        
        if self.use_kalshi and self._kalshi_client:
            self._load_kalshi_trends()
        if not self.llm.health_check():
            logger.warning(
                "Ollama not responding. Ensure 'ollama serve' is running."
            )
        
        personas = self.custom_agents or self._load_personas()
        self._create_agents(personas)
        self.seed_tweets = self._load_seed_tweets()
        
        logger.info(f"Setup complete: {len(self.agents)} agents created")
    
    def run(self) -> list[dict[str, Any]]:
        """Execute the main simulation loop.
        
        Returns:
            List of action result dictionaries.
        """
        if not self.agents:
            self.setup()
        
        total_steps = self.days * STEPS_PER_DAY
        
        logger.info(f"Starting simulation: {total_steps} time steps")
        
        with tqdm(total=total_steps, desc="Simulating", unit="step") as pbar:
            for step in range(total_steps):
                if self.stop_requested:
                    logger.info("Simulation stopping early due to user request")
                    break
                
                self._execute_step(step)
                pbar.update(1)
                pbar.set_postfix({
                    "day": step // STEPS_PER_DAY + 1,
                    "price": f"${self._current_price:.2f}",
                    "actions": len(self.simulation_log),
                })
        
        self._save_results()
        
        logger.info(
            f"Simulation complete: {len(self.simulation_log)} actions recorded"
        )
        
        return self.simulation_log
    
    def _execute_step(self, step: int) -> None:
        """Execute a single simulation time step.
        
        Args:
            step: Current step number.
        """
        self._current_time = self.START_DATE + timedelta(hours=step * TIME_STEP_HOURS)
        self._current_step_index = step
        
        self._update_market_state(step)
        
        market_info = self._get_market_info()
        social_info = self._get_social_info()
        
        step_actions: list[ActionResult] = []
        
        active_agents = random.sample(
            self.agents, 
            k=min(len(self.agents), max(5, len(self.agents) // 3))
        )
        
        for agent in active_agents:
            agent.observe(market_info, social_info)
            
            decision = agent.decide(self.llm)
            
            action = agent.act(decision)
            step_actions.append(action)
            
            self.simulation_log.append(action.to_dict())
        
        self._update_community_sentiment(step_actions)
    
    def _update_market_state(self, step: int) -> None:
        """Update simulated market conditions.
        
        Args:
            step: Current step number for price trajectory.
        """
        day = step // STEPS_PER_DAY
        hour_of_day = step % STEPS_PER_DAY
        
        base_multiplier = 1.0
        if day < 3:
            base_multiplier = 1.0 + (day * 0.1)
        elif day < 5:
            base_multiplier = 1.3 + ((day - 3) * 0.5)
        else:
            base_multiplier = 2.3 + ((day - 5) * 2.0)
        
        sentiment_boost = 1.0 + (self._community_sentiment * 0.1)
        
        noise = random.gauss(0, 0.02)
        
        self._current_price = self.BASE_PRICE * base_multiplier * sentiment_boost * (1 + noise)
        self._current_price = max(self._current_price, 1.0)
        
        self._price_history.append(self._current_price)
    
    def _get_market_info(self) -> MarketInfo:
        """Get current market information.
        
        Returns:
            MarketInfo with current market state.
        """
        prev_price = self._price_history[-2] if len(self._price_history) > 1 else self.BASE_PRICE
        change_pct = ((self._current_price - prev_price) / prev_price) * 100
        
        if change_pct > 5:
            trend = "surging"
        elif change_pct > 1:
            trend = "rising"
        elif change_pct < -5:
            trend = "crashing"
        elif change_pct < -1:
            trend = "falling"
        else:
            trend = "stable"
        
        return MarketInfo(
            timestamp=self._current_time,
            stock_price=round(self._current_price, 2),
            price_change_pct=round(change_pct, 2),
            volume=random.randint(1000000, 50000000),
            trend=trend,
        )
    
    def _load_kalshi_trends(self) -> None:
        """Load trending topics from Kalshi API."""
        if not self._kalshi_client:
            return
        
        try:
            events = self._kalshi_client.get_trending_events(limit=10)
            self._kalshi_analysis = self._kalshi_client.analyze_trends(events)
            topics = self._kalshi_analysis.get("topics", [])
            logger.info(f"Loaded {len(topics)} trending topics from Kalshi")
        except Exception as e:
            logger.warning(f"Failed to load Kalshi trends: {e}")
            self._kalshi_analysis = None

    def _get_social_info(self) -> SocialMediaInfo:
        """Get current social media environment.
        
        Returns:
            SocialMediaInfo with current social context.
        """
        if self.use_kalshi and self._kalshi_analysis:
            kalshi_topics = self._kalshi_analysis.get("topics", [])[:5]
            trending = kalshi_topics if kalshi_topics else ["$GME", "GameStop", "WallStreetBets"]
        else:
            trending = ["$GME", "GameStop", "WallStreetBets"]
        
        if self._community_sentiment > 0.3:
            trending.extend(["diamondhands", "tothemoon", "HOLD"])
        elif self._community_sentiment < -0.3:
            trending.extend(["sell", "crash", "paperhands"])
        else:
            trending.extend(["stocks", "trading", "investing"])
        
        sample_size = min(5, len(self.seed_tweets)) if self.seed_tweets else 0
        sample_tweets = random.sample(self.seed_tweets, sample_size) if sample_size > 0 else []
        
        recent_agent_tweets = [
            log["content"] 
            for log in self.simulation_log[-20:] 
            if log.get("action_type") == "TWEET"
        ]
        sample_tweets.extend(recent_agent_tweets[-3:])
        
        return SocialMediaInfo(
            timestamp=self._current_time,
            trending_topics=trending,
            sample_tweets=sample_tweets,
            sentiment_score=round(self._community_sentiment, 2),
        )
    
    def _update_community_sentiment(self, actions: list[ActionResult]) -> None:
        """Update community sentiment based on agent actions.
        
        Args:
            actions: List of actions from current step.
        """
        positive_keywords = {"moon", "hold", "diamond", "buy", "bullish", "rocket", "ape"}
        negative_keywords = {"sell", "crash", "dump", "paper", "fear", "loss"}
        
        sentiment_delta = 0.0
        tweet_count = 0
        
        for action in actions:
            if action.action_type == "TWEET":
                tweet_count += 1
                content_lower = action.content.lower()
                
                pos_count = sum(1 for kw in positive_keywords if kw in content_lower)
                neg_count = sum(1 for kw in negative_keywords if kw in content_lower)
                
                sentiment_delta += (pos_count - neg_count) * 0.05
        
        if tweet_count > 0:
            self._community_sentiment += sentiment_delta / tweet_count
        
        self._community_sentiment *= 0.95
        self._community_sentiment = max(-1.0, min(1.0, self._community_sentiment))
    
    def _load_personas(self) -> list[dict[str, Any]]:
        """Load personas with priority chain: SocioVerse > custom > Kalshi-generated > file > defaults.
        
        Returns:
            List of persona dictionaries.
        """
        # Priority 1: Try SocioVerse dataset (real user data) - HIGHEST PRIORITY
        if self.use_kalshi:
            personas = self._try_load_socioverse()
            if personas:
                return personas
        
        # Priority 2: Custom agents provided via API request
        if self.custom_agents:
            logger.info(f"Using {len(self.custom_agents)} custom agents from request")
            return self.custom_agents
        
        # Priority 3: Generate personas from Kalshi trends using LLM
        if self.use_kalshi and self._kalshi_analysis:
            personas = self._try_generate_from_kalshi()
            if personas:
                return personas
        
        # Priority 4: Generate defaults (static file loading deprecated)
        logger.info("Generating default personas")
        return self._generate_default_personas()

    def _try_load_socioverse(self) -> list[dict[str, Any]] | None:
        """Attempt to load personas from SocioVerse dataset.
        
        Returns:
            List of personas or None if unavailable.
        """
        try:
            if self._user_pool_provider is not None:
                personas = self._user_pool_provider.fetch_user_pool(count=self.agent_count)
            else:
                from .socioverse_connector import SocioVerseConnector
                connector = SocioVerseConnector()
                personas = connector.fetch_user_pool(count=self.agent_count)
            
            if personas:
                logger.info(f"Loaded {len(personas)} personas from SocioVerse")
                return personas
            
        except ImportError:
            logger.debug("SocioVerse connector not available")
        except Exception as e:
            logger.warning(f"SocioVerse fetch failed: {e}")
        
        return None

    def _try_generate_from_kalshi(self) -> list[dict[str, Any]] | None:
        """Attempt to generate personas from Kalshi trends using LLM.
        
        Returns:
            List of personas or None if generation fails.
        """
        try:
            trend_summary = self._kalshi_analysis.get("summary", "")
            if not trend_summary:
                return None
            
            if not self.llm:
                self.llm = LlamaInterface(mock_mode=self.mock_llm)
            
            from .agent_generator import AgentGenerator
            
            generator = AgentGenerator(self.llm)
            personas = generator.generate_agents(
                trend_summary=trend_summary,
                count=self.agent_count,
            )
            
            if personas:
                logger.info(f"Generated {len(personas)} personas from Kalshi trends")
                return personas
            
        except Exception as e:
            logger.warning(f"Kalshi persona generation failed: {e}")
        
        return None
    
    def _generate_default_personas(self) -> list[dict[str, Any]]:
        """Generate default personas if file not available.
        
        Returns:
            List of generated persona dictionaries.
        """
        personality_types = [
            ["risk-seeking", "impulsive", "optimistic"],
            ["cautious", "analytical", "skeptical"],
            ["trend-following", "social", "enthusiastic"],
            ["contrarian", "independent", "patient"],
            ["emotional", "reactive", "community-focused"],
        ]
        
        interest_sets = [
            ["stocks", "crypto", "reddit"],
            ["gaming", "memes", "investing"],
            ["finance", "technology", "social media"],
            ["trading", "entertainment", "news"],
        ]
        
        personas = []
        for i in range(self.agent_count):
            personas.append({
                "id": i,
                "name": f"User_{i:03d}",
                "personality_traits": random.choice(personality_types),
                "interests": random.choice(interest_sets),
                "beliefs": {
                    "risk_tolerance": random.choice(["low", "moderate", "high"]),
                    "market_outlook": random.choice(["bullish", "neutral", "bearish"]),
                    "trust_in_institutions": random.choice(["low", "moderate", "high"]),
                },
                "social": {
                    "follower_count": random.randint(10, 10000),
                    "influence_score": random.random(),
                },
            })
        
        return personas
    
    def _create_agents(self, personas: list[dict[str, Any]]) -> None:
        """Create agent instances from personas.
        
        Args:
            personas: List of persona dictionaries.
        """
        for i in range(self.agent_count):
            persona = personas[i % len(personas)] if personas else {}
            agent = Agent(agent_id=i, persona=persona, market_topic=self.market_topic)
            self.agents.append(agent)
    
    def _load_seed_tweets(self) -> list[str]:
        """Load seed content, preferring dynamic Kalshi-based content.
        
        Returns:
            List of tweet/content strings.
        """
        # Priority 1: Generate dynamic content from Kalshi trends
        if self.use_kalshi and self._kalshi_analysis:
            topics = self._kalshi_analysis.get("topics", [])
            if topics:
                dynamic_content = self._generate_kalshi_based_content(topics)
                if dynamic_content:
                    logger.info(f"Generated {len(dynamic_content)} dynamic seed posts from Kalshi topics")
                    return dynamic_content
        
        # Priority 2: Generate sample content (static file loading deprecated)
        return self._generate_sample_tweets()

    def _generate_kalshi_based_content(self, topics: list[str]) -> list[str]:
        """Generate seed content based on Kalshi trending topics.
        
        Args:
            topics: List of trending topic strings from Kalshi.
            
        Returns:
            List of generated content strings.
        """
        content = []
        
        templates = [
            "What do you think about {topic}?",
            "Just saw the latest on {topic}. Thoughts?",
            "The odds on {topic} are interesting. Anyone tracking this?",
            "{topic} - this could be huge!",
            "Keeping an eye on {topic}. Market sentiment looks strong.",
            "Does anyone have insight on {topic}?",
            "The prediction market for {topic} is heating up!",
            "I'm bullish on {topic}. Here's why...",
            "Skeptical about {topic}. What am I missing?",
            "{topic} - make your bets now or regret later!",
        ]
        
        for topic in topics[:10]:
            for template in templates:
                content.append(template.format(topic=topic))
        
        return content
    
    def _generate_sample_tweets(self) -> list[str]:
        """Generate sample tweets for simulation.
        
        Returns:
            List of sample tweet strings.
        """
        return [
            "$GME to the moon! 🚀🚀🚀 Diamond hands forever!",
            "Just bought more GME. Holding strong! 💎🙌",
            "The shorts haven't covered yet. This is just the beginning!",
            "GameStop is the future. Don't let the hedge funds win!",
            "Paper hands selling already? We're just getting started!",
            "HOLD THE LINE! Apes together strong! 🦍",
            "Not financial advice but I like the stock $GME",
            "Watching GME charts all day. Can't look away!",
            "The squeeze hasn't squoze yet. Stay strong!",
            "Robinhood halting buys? This is manipulation!",
            "I'm not selling until we hit $1000. Diamond hands!",
            "My wife's boyfriend said to buy more GME",
            "Tendies incoming! 🍗🍗🍗",
            "This is bigger than money. It's about sending a message.",
            "WSB vs Wall Street. History in the making!",
        ]
    
    def _save_results(self) -> None:
        """Save simulation results to file."""
        output_data = {
            "metadata": {
                "simulation_days": self.days,
                "agent_count": self.agent_count,
                "total_steps": self.days * STEPS_PER_DAY,
                "start_time": self.START_DATE.isoformat(),
                "end_time": self._current_time.isoformat(),
                "total_actions": len(self.simulation_log),
            },
            "price_history": self._price_history,
            "actions": self.simulation_log,
        }
        
        try:
            with open(SIMULATION_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {SIMULATION_LOG_FILE}")
        except IOError as e:
            logger.error(f"Failed to save results: {e}")
            raise SimulationError(f"Failed to save results: {e}") from e
