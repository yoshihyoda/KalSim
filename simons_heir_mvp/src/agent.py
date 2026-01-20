"""Agent module defining simulation participants with 7-layer model integration.

Contains the Agent class that models individual actors using the full
7-layer model for cognitive, emotional, and social processing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm_interface import LlamaInterface
from .layers.layer1_neurobiology import NeurobiologyModule, NeurobiologicalState
from .layers.layer2_cognition import CognitionModule, CognitiveBiases
from .layers.layer3_emotion import EmotionModule, EmotionState
from .layers.layer4_social_interaction import SocialInteractionModule
from .layers.layer5_collective_identity import IdentityModule, IdentityState, IdentityGroup

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """Container for market state information."""
    timestamp: datetime
    stock_price: float
    price_change_pct: float
    volume: int
    trend: str


@dataclass
class SocialMediaInfo:
    """Container for social media environment information."""
    timestamp: datetime
    trending_topics: list[str]
    sample_tweets: list[str]
    sentiment_score: float


@dataclass
class MemoryEntry:
    """Single entry in agent's memory."""
    timestamp: datetime
    entry_type: str
    content: dict[str, Any]


@dataclass
class ActionResult:
    """Result of an agent action."""
    agent_id: int
    timestamp: datetime
    action_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class AgentState:
    """Aggregated state from all 7 layers.
    
    Provides a unified view of the agent's internal state
    across neurobiological, cognitive, emotional, social, and identity dimensions.
    """
    neurobiological: NeurobiologicalState = field(default_factory=NeurobiologicalState)
    cognitive: CognitiveBiases = field(default_factory=CognitiveBiases)
    emotional: EmotionState = field(default_factory=EmotionState)
    identity: IdentityState | None = None
    
    social_pressure: float = 0.0
    herding_detected: bool = False
    viral_exposure: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "fomo_level": self.neurobiological.fomo_level,
            "dopamine_response": self.neurobiological.dopamine_response,
            "stress_level": self.neurobiological.stress_level,
            "social_proof": self.cognitive.social_proof,
            "confirmation_bias": self.cognitive.confirmation_bias,
            "loss_aversion": self.cognitive.loss_aversion,
            "valence": self.emotional.valence,
            "arousal": self.emotional.arousal,
            "dominant_emotion": self.emotional.dominant_emotion,
            "identity_group": self.identity.primary_group.name if self.identity else "NEUTRAL",
            "group_identification": self.identity.group_identification if self.identity else 0.0,
            "social_pressure": self.social_pressure,
            "herding_detected": self.herding_detected,
            "viral_exposure": self.viral_exposure,
        }


class Agent:
    """Simulation agent with 7-layer model integration.
    
    Each agent processes stimuli through the 7-layer model pipeline,
    maintaining rich internal state that influences LLM-based decisions.
    """
    
    MEMORY_LIMIT: int = 50
    
    def __init__(self, agent_id: int, persona: dict[str, Any], market_topic: str = "prediction markets") -> None:
        """Initialize an Agent with 7-layer modules.
        
        Args:
            agent_id: Unique identifier for this agent.
            persona: Dictionary containing agent's personality profile.
            market_topic: The market topic this agent will discuss.
        """
        self.agent_id = agent_id
        self.persona = persona
        self.market_topic = market_topic
        self.memory: list[MemoryEntry] = []
        self._current_timestamp: datetime = datetime.now()
        
        self._neuro_module = NeurobiologyModule()
        self._cognition_module = CognitionModule()
        self._emotion_module = EmotionModule()
        self._social_module = SocialInteractionModule()
        self._identity_module = IdentityModule()
        
        self.state = AgentState()
        
        self.state.identity = self._identity_module.assign_identity(persona)
    
    @property
    def name(self) -> str:
        """Get agent's display name from persona."""
        return self.persona.get("name", f"Agent_{self.agent_id}")
    
    @property
    def identity_group(self) -> str:
        """Get agent's identity group."""
        if self.state.identity:
            return self.state.identity.primary_group.name
        return "NEUTRAL"
    
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
        
        observation = {"market": None, "social": None}
        
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

    def update_layer_states(
        self,
        market_info: MarketInfo | None,
        social_media_info: SocialMediaInfo | None,
    ) -> None:
        """Update all layer states based on observations.
        
        This runs the full 7-layer pipeline to update internal state.
        
        Args:
            market_info: Market information.
            social_media_info: Social media information.
        """
        market_state = {}
        if market_info:
            market_state = {
                "price_change_pct": market_info.price_change_pct,
                "trend": market_info.trend,
                "volatility": abs(market_info.price_change_pct) / 100,
            }
        
        social_state = {}
        if social_media_info:
            social_state = {
                "sentiment": social_media_info.sentiment_score,
                "viral_posts": [],
            }
        
        neuro_input = {
            "agent": {
                "neuro_state": {
                    "fomo_level": self.state.neurobiological.fomo_level,
                    "dopamine_response": self.state.neurobiological.dopamine_response,
                    "stress_level": self.state.neurobiological.stress_level,
                    "reward_sensitivity": self.state.neurobiological.reward_sensitivity,
                    "habituation": self.state.neurobiological.habituation,
                },
                "portfolio": {},
            },
            "market": market_state,
            "social": social_state,
        }
        neuro_output = self._neuro_module.process(neuro_input)
        
        self.state.neurobiological = NeurobiologicalState(
            fomo_level=neuro_output.get("fomo_level", 0.0),
            dopamine_response=neuro_output.get("dopamine_response", 0.5),
            stress_level=neuro_output.get("stress_level", 0.0),
            reward_sensitivity=neuro_output.get("reward_sensitivity", 0.5),
            habituation=neuro_output.get("habituation", 0.0),
        )
        
        stimulus_type = None
        stimulus_intensity = 0.0
        if market_info:
            if market_info.trend == "surging":
                stimulus_type = "market_surge"
                stimulus_intensity = min(abs(market_info.price_change_pct) / 50, 1.0)
            elif market_info.trend == "crashing":
                stimulus_type = "market_crash"
                stimulus_intensity = min(abs(market_info.price_change_pct) / 50, 1.0)
        
        emotion_input = {
            "agent": {
                "emotion": {
                    "valence": self.state.emotional.valence,
                    "arousal": self.state.emotional.arousal,
                },
            },
            "stimulus": {
                "type": stimulus_type,
                "intensity": stimulus_intensity,
            } if stimulus_type else {},
        }
        emotion_output = self._emotion_module.process(emotion_input)
        
        self.state.emotional = EmotionState(
            valence=emotion_output.get("valence", 0.0),
            arousal=emotion_output.get("arousal", 0.5),
            dominant_emotion=emotion_output.get("dominant_emotion", "neutral"),
            intensity=emotion_output.get("emotion_intensity", 0.5),
        )
        
        cognition_input = {
            "agent": {
                "beliefs": self.persona.get("beliefs", {}),
            },
            "social": {
                "consensus_view": "bullish" if social_state.get("sentiment", 0) > 0.3 else "neutral",
                "consensus_strength": abs(social_state.get("sentiment", 0)),
                "peer_count": 50,
            },
            "market": market_state,
            "information": {},
        }
        self._cognition_module.process(cognition_input)
        
        identity_output = self._identity_module.process({
            "agent": {"persona": self.persona},
            "social": {"dominant_group": "WSB" if social_state.get("sentiment", 0) > 0.5 else ""},
        })
        
        self.state.social_pressure = 0.0
        self.state.herding_detected = False
        self.state.viral_exposure = False
    
    def decide(self, llm_interface: LlamaInterface) -> str:
        """Make a decision using LLM with layer-informed context.
        
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
            fallback = "ACTION: HOLD\nCONTENT: Unable to make decision at this time."
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
                "layer_state": self.state.to_dict(),
            },
        )
        
        self._add_memory("action", result.to_dict())
        logger.debug(f"Agent {self.agent_id} acted: {action_type}")
        
        return result
    
    def _build_decision_prompt(self) -> str:
        """Build the prompt with layer-informed context.
        
        Returns:
            Formatted prompt string including psychological state.
        """
        recent_memories = self._get_recent_memories(5)
        memory_context = self._format_memories(recent_memories)
        
        layer_context = self._build_layer_context()
        
        prompt = f"""You are simulating a social media user discussing prediction markets.

TOPIC: {self.market_topic}

CHARACTER PROFILE:
- Name: {self.name}
- {self.personality_summary}
- Identity Group: {self.identity_group}

PSYCHOLOGICAL STATE:
{layer_context}

RECENT CONTEXT:
{memory_context}

Based on your character, psychological state, and the current situation, decide what to do next.
Choose ONE action and provide your response in this exact format:

ACTION: [TWEET/HOLD/LURK]
CONTENT: [If TWEET, write a short post (max 280 chars) about "{self.market_topic}". If HOLD or LURK, briefly explain why.]

Remember to stay in character and let your psychological state influence your decision."""

        return prompt
    
    def _build_layer_context(self) -> str:
        """Build context string from layer states.
        
        Returns:
            Formatted psychological state context.
        """
        context_parts = []
        
        fomo = self.state.neurobiological.fomo_level
        if fomo > 0.7:
            context_parts.append(f"- You are experiencing strong FOMO (level: {fomo:.1f})")
        elif fomo > 0.4:
            context_parts.append(f"- You feel moderate fear of missing out (level: {fomo:.1f})")
        
        stress = self.state.neurobiological.stress_level
        if stress > 0.7:
            context_parts.append(f"- You are highly stressed (level: {stress:.1f})")
        elif stress > 0.4:
            context_parts.append(f"- You feel some stress (level: {stress:.1f})")
        
        emotion = self.state.emotional.dominant_emotion
        intensity = self.state.emotional.intensity
        if emotion != "neutral":
            context_parts.append(f"- Dominant emotion: {emotion} (intensity: {intensity:.1f})")
        
        if self.state.identity:
            group = self.state.identity.primary_group.name
            identification = self.state.identity.group_identification
            if identification > 0.6:
                context_parts.append(f"- You strongly identify with {group} (strength: {identification:.1f})")
        
        if self.state.herding_detected:
            context_parts.append("- You notice everyone around you taking similar actions")
        
        if self.state.viral_exposure:
            context_parts.append("- You've seen viral posts energizing the community")
        
        return "\n".join(context_parts) if context_parts else "- No significant psychological factors at play."
    
    def _parse_decision(self, decision: str) -> tuple[str, str]:
        """Parse LLM decision into action type and content."""
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
        """Add an entry to agent's memory with size limit."""
        entry = MemoryEntry(
            timestamp=self._current_timestamp,
            entry_type=entry_type,
            content=content,
        )
        self.memory.append(entry)
        
        if len(self.memory) > self.MEMORY_LIMIT:
            self.memory = self.memory[-self.MEMORY_LIMIT:]
    
    def _get_recent_memories(self, count: int) -> list[MemoryEntry]:
        """Get the most recent memory entries."""
        return self.memory[-count:] if self.memory else []
    
    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories into a readable string for prompts."""
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
    
    def get_layer_summary(self) -> str:
        """Get a summary of all layer states.
        
        Returns:
            Formatted summary string.
        """
        return (
            f"Neuro: FOMO={self.state.neurobiological.fomo_level:.2f}, "
            f"Stress={self.state.neurobiological.stress_level:.2f} | "
            f"Emotion: {self.state.emotional.dominant_emotion} | "
            f"Identity: {self.identity_group}"
        )
    
    def reset_layers(self) -> None:
        """Reset all layer modules to initial state."""
        self._neuro_module.reset()
        self._cognition_module.reset()
        self._emotion_module.reset()
        self._social_module.reset()
        self._identity_module.reset()
        
        self.state = AgentState()
        self.state.identity = self._identity_module.assign_identity(self.persona)
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"Agent(id={self.agent_id}, name='{self.name}', group='{self.identity_group}')"
