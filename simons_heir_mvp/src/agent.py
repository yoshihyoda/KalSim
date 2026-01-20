"""Agent module defining simulation participants.

Contains the Agent class that models individual actors in the social simulation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm_interface import LlamaInterface

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """Container for market state information.
    
    Attributes:
        timestamp: Current simulation time.
        stock_price: Current stock price.
        price_change_pct: Percentage change from previous step.
        volume: Trading volume.
        trend: Overall market trend description.
    """
    timestamp: datetime
    stock_price: float
    price_change_pct: float
    volume: int
    trend: str


@dataclass
class SocialMediaInfo:
    """Container for social media environment information.
    
    Attributes:
        timestamp: Current simulation time.
        trending_topics: List of trending hashtags/topics.
        sample_tweets: Sample of recent tweets visible to agent.
        sentiment_score: Overall community sentiment (-1 to 1).
    """
    timestamp: datetime
    trending_topics: list[str]
    sample_tweets: list[str]
    sentiment_score: float


@dataclass
class MemoryEntry:
    """Single entry in agent's memory.
    
    Attributes:
        timestamp: When the memory was created.
        entry_type: Type of memory (observation, decision, action).
        content: The memory content.
    """
    timestamp: datetime
    entry_type: str
    content: dict[str, Any]


@dataclass
class ActionResult:
    """Result of an agent action.
    
    Attributes:
        agent_id: ID of the acting agent.
        timestamp: When the action occurred.
        action_type: Type of action taken.
        content: Action content (e.g., tweet text).
        metadata: Additional action metadata.
    """
    agent_id: int
    timestamp: datetime
    action_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "content": self.content,
            "metadata": self.metadata,
        }


class Agent:
    """Simulation agent representing a social media user.
    
    Each agent has a unique persona, maintains memory of observations and actions,
    and makes decisions using the LLM based on their personality and context.
    
    Attributes:
        agent_id: Unique identifier for the agent.
        persona: Dictionary containing personality traits, interests, and beliefs.
        memory: List of past observations and actions.
    """
    
    MEMORY_LIMIT: int = 50
    
    def __init__(self, agent_id: int, persona: dict[str, Any]) -> None:
        """Initialize an Agent.
        
        Args:
            agent_id: Unique identifier for this agent.
            persona: Dictionary containing agent's personality profile.
        """
        self.agent_id = agent_id
        self.persona = persona
        self.memory: list[MemoryEntry] = []
        self._current_timestamp: datetime = datetime.now()
    
    @property
    def name(self) -> str:
        """Get agent's display name from persona."""
        return self.persona.get("name", f"Agent_{self.agent_id}")
    
    @property
    def personality_summary(self) -> str:
        """Get a summary of agent's personality for prompts."""
        traits = self.persona.get("personality_traits", [])
        interests = self.persona.get("interests", [])
        beliefs = self.persona.get("beliefs", {})
        
        summary_parts = []
        if traits:
            summary_parts.append(f"Personality: {', '.join(traits[:3])}")
        if interests:
            summary_parts.append(f"Interests: {', '.join(interests[:3])}")
        if beliefs:
            risk_tolerance = beliefs.get("risk_tolerance", "moderate")
            summary_parts.append(f"Risk tolerance: {risk_tolerance}")
        
        return "; ".join(summary_parts) if summary_parts else "Average user"
    
    def observe(
        self,
        market_info: MarketInfo | None,
        social_media_info: SocialMediaInfo | None,
    ) -> None:
        """Observe and internalize market and social media information.
        
        Args:
            market_info: Current market state, or None if unavailable.
            social_media_info: Current social media environment, or None.
        """
        if market_info:
            self._current_timestamp = market_info.timestamp
        elif social_media_info:
            self._current_timestamp = social_media_info.timestamp
        
        observation = {
            "market": None,
            "social": None,
        }
        
        if market_info:
            observation["market"] = {
                "price": market_info.stock_price,
                "change_pct": market_info.price_change_pct,
                "trend": market_info.trend,
            }
        
        if social_media_info:
            observation["social"] = {
                "trending": social_media_info.trending_topics[:5],
                "tweets_seen": len(social_media_info.sample_tweets),
                "community_sentiment": social_media_info.sentiment_score,
            }
        
        self._add_memory("observation", observation)
        logger.debug(f"Agent {self.agent_id} observed: {observation}")
    
    def decide(self, llm_interface: LlamaInterface) -> str:
        """Make a decision about next action using LLM.
        
        Args:
            llm_interface: Interface to the LLM for decision generation.
            
        Returns:
            Decision string from the LLM.
        """
        prompt = self._build_decision_prompt()
        
        try:
            decision = llm_interface.generate(prompt, temperature=0.8)
            self._add_memory("decision", {"raw_decision": decision})
            logger.debug(f"Agent {self.agent_id} decided: {decision[:100]}...")
            return decision
        except Exception as e:
            logger.error(f"Agent {self.agent_id} decision failed: {e}")
            fallback = "HOLD - Unable to make decision at this time."
            self._add_memory("decision", {"raw_decision": fallback, "error": str(e)})
            return fallback
    
    def act(self, decision: str) -> ActionResult:
        """Execute an action based on the decision.
        
        Args:
            decision: The decision string from the LLM.
            
        Returns:
            ActionResult containing the action details.
        """
        action_type, content = self._parse_decision(decision)
        
        result = ActionResult(
            agent_id=self.agent_id,
            timestamp=self._current_timestamp,
            action_type=action_type,
            content=content,
            metadata={
                "persona_name": self.name,
                "personality": self.personality_summary,
            },
        )
        
        self._add_memory("action", result.to_dict())
        logger.debug(f"Agent {self.agent_id} acted: {action_type}")
        
        return result
    
    def _build_decision_prompt(self) -> str:
        """Build the prompt for LLM decision-making.
        
        Returns:
            Formatted prompt string.
        """
        recent_memories = self._get_recent_memories(5)
        memory_context = self._format_memories(recent_memories)
        
        prompt = f"""You are simulating a social media user during the GameStop stock surge in January 2021.

CHARACTER PROFILE:
- Name: {self.name}
- {self.personality_summary}

RECENT CONTEXT:
{memory_context}

Based on your character and the current situation, decide what to do next.
Choose ONE action and provide your response in this exact format:

ACTION: [TWEET/HOLD/LURK]
CONTENT: [If TWEET, write a short tweet (max 280 chars) about GME. If HOLD or LURK, briefly explain why.]

Remember to stay in character and respond authentically based on your personality."""

        return prompt
    
    def _parse_decision(self, decision: str) -> tuple[str, str]:
        """Parse LLM decision into action type and content.
        
        Args:
            decision: Raw decision string from LLM.
            
        Returns:
            Tuple of (action_type, content).
        """
        lines = decision.strip().split("\n")
        action_type = "LURK"
        content = ""
        
        for line in lines:
            line_upper = line.upper()
            if line_upper.startswith("ACTION:"):
                action_part = line.split(":", 1)[1].strip().upper()
                if "TWEET" in action_part:
                    action_type = "TWEET"
                elif "HOLD" in action_part:
                    action_type = "HOLD"
                else:
                    action_type = "LURK"
            elif line_upper.startswith("CONTENT:"):
                content = line.split(":", 1)[1].strip()
        
        if not content and action_type == "TWEET":
            content = decision[:280]
        
        return action_type, content[:280] if content else "No specific action taken."
    
    def _add_memory(self, entry_type: str, content: dict[str, Any]) -> None:
        """Add an entry to agent's memory with size limit.
        
        Args:
            entry_type: Type of memory entry.
            content: Memory content dictionary.
        """
        entry = MemoryEntry(
            timestamp=self._current_timestamp,
            entry_type=entry_type,
            content=content,
        )
        self.memory.append(entry)
        
        if len(self.memory) > self.MEMORY_LIMIT:
            self.memory = self.memory[-self.MEMORY_LIMIT:]
    
    def _get_recent_memories(self, count: int) -> list[MemoryEntry]:
        """Get the most recent memory entries.
        
        Args:
            count: Number of recent memories to retrieve.
            
        Returns:
            List of recent MemoryEntry objects.
        """
        return self.memory[-count:] if self.memory else []
    
    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories into a readable string for prompts.
        
        Args:
            memories: List of memory entries to format.
            
        Returns:
            Formatted string representation of memories.
        """
        if not memories:
            return "No recent activity."
        
        formatted_lines = []
        for mem in memories:
            time_str = mem.timestamp.strftime("%H:%M")
            match mem.entry_type:
                case "observation":
                    market = mem.content.get("market", {})
                    social = mem.content.get("social", {})
                    if market:
                        formatted_lines.append(
                            f"[{time_str}] Market: ${market.get('price', 'N/A')}, "
                            f"{market.get('change_pct', 0):+.1f}%, {market.get('trend', 'stable')}"
                        )
                    if social:
                        formatted_lines.append(
                            f"[{time_str}] Social: Sentiment {social.get('community_sentiment', 0):.2f}, "
                            f"Trending: {', '.join(social.get('trending', [])[:3])}"
                        )
                case "action":
                    formatted_lines.append(
                        f"[{time_str}] You: {mem.content.get('action_type', 'unknown')} - "
                        f"{mem.content.get('content', '')[:50]}..."
                    )
        
        return "\n".join(formatted_lines) if formatted_lines else "No significant memories."
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"Agent(id={self.agent_id}, name='{self.name}')"
