"""Dynamic Prompt Builder for Agent Decision-Making.

Generates context-aware prompts based on the full 7-layer model state,
translating psychological states into natural language that guides LLM behavior.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds dynamic prompts from agent and layer states.
    
    Translates numerical psychological states from the 7-layer model
    into natural language context that influences LLM decision-making.
    
    Attributes:
        agent_state: Current agent state dictionary.
        layer_outputs: Outputs from BehaviorEngine processing.
    """
    
    FOMO_HIGH_THRESHOLD = 0.7
    FOMO_MODERATE_THRESHOLD = 0.4
    STRESS_HIGH_THRESHOLD = 0.7
    STRESS_MODERATE_THRESHOLD = 0.4
    AROUSAL_HIGH_THRESHOLD = 0.7
    SOCIAL_PRESSURE_THRESHOLD = 0.5
    GROUP_IDENTIFICATION_THRESHOLD = 0.6
    
    def __init__(
        self,
        agent_state: dict[str, Any],
        layer_outputs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the PromptBuilder.
        
        Args:
            agent_state: Dictionary containing agent's current state.
            layer_outputs: Optional outputs from BehaviorEngine processing.
        """
        self.agent_state = agent_state
        self.layer_outputs = layer_outputs or {}
    
    def build_decision_prompt(
        self,
        market_topic: str = "prediction markets",
        recent_context: str = "No recent activity.",
    ) -> str:
        """Build a complete decision prompt for the LLM.
        
        Args:
            market_topic: The topic being discussed.
            recent_context: Formatted recent memory/context string.
            
        Returns:
            Complete prompt string for LLM decision generation.
        """
        character_profile = self._build_character_profile()
        psychological_state = self._build_psychological_state()
        action_guidance = self._build_action_guidance()
        
        prompt = f"""You are simulating a social media user discussing {market_topic}.

TOPIC: {market_topic}

{character_profile}

{psychological_state}

RECENT CONTEXT:
{recent_context}

{action_guidance}

Based on your character, psychological state, and the current situation, decide what to do next.
Choose ONE action and provide your response in this exact format:

ACTION: [TWEET/HOLD/LURK]
CONTENT: [If TWEET, write a short post (max 280 chars) about "{market_topic}". If HOLD or LURK, briefly explain why.]

Remember to stay in character and let your psychological state influence your decision."""

        return prompt
    
    def _build_character_profile(self) -> str:
        """Build the character profile section of the prompt.
        
        Returns:
            Formatted character profile string.
        """
        name = self.agent_state.get("name", "Anonymous")
        personality = self.agent_state.get("personality_summary", "Average user")
        identity_group = self.agent_state.get("identity_group", "NEUTRAL")
        
        return f"""CHARACTER PROFILE:
- Name: {name}
- {personality}
- Identity Group: {identity_group}"""
    
    def _build_psychological_state(self) -> str:
        """Build the psychological state section based on layer outputs.
        
        Translates numerical states into descriptive natural language.
        
        Returns:
            Formatted psychological state string.
        """
        context_parts = []
        
        fomo_context = self._get_fomo_context()
        if fomo_context:
            context_parts.append(fomo_context)
        
        stress_context = self._get_stress_context()
        if stress_context:
            context_parts.append(stress_context)
        
        emotion_context = self._get_emotion_context()
        if emotion_context:
            context_parts.append(emotion_context)
        
        social_context = self._get_social_context()
        if social_context:
            context_parts.append(social_context)
        
        identity_context = self._get_identity_context()
        if identity_context:
            context_parts.append(identity_context)
        
        cognitive_context = self._get_cognitive_context()
        if cognitive_context:
            context_parts.append(cognitive_context)
        
        if not context_parts:
            context_parts.append("- You feel calm and analytical about the current situation.")
        
        return "PSYCHOLOGICAL STATE:\n" + "\n".join(context_parts)
    
    def _get_fomo_context(self) -> str | None:
        """Get FOMO-related context from Layer 1 (Neurobiology).
        
        Returns:
            FOMO context string or None.
        """
        fomo = self.layer_outputs.get("fomo_level", 0.0)
        
        if fomo > self.FOMO_HIGH_THRESHOLD:
            return (
                f"- URGENT: You are experiencing intense FOMO (level: {fomo:.1f}). "
                "Everyone seems to be making money and you feel like you're being left behind! "
                "The urge to act NOW is overwhelming."
            )
        elif fomo > self.FOMO_MODERATE_THRESHOLD:
            return (
                f"- You feel moderate fear of missing out (level: {fomo:.1f}). "
                "You notice others taking action and wonder if you should too."
            )
        return None
    
    def _get_stress_context(self) -> str | None:
        """Get stress-related context from Layer 1 (Neurobiology).
        
        Returns:
            Stress context string or None.
        """
        stress = self.layer_outputs.get("stress_level", 0.0)
        
        if stress > self.STRESS_HIGH_THRESHOLD:
            return (
                f"- You are highly stressed (level: {stress:.1f}). "
                "Your heart is racing and it's hard to think clearly. "
                "You feel pressure to do something."
            )
        elif stress > self.STRESS_MODERATE_THRESHOLD:
            return (
                f"- You feel some stress and tension (level: {stress:.1f}). "
                "The situation is weighing on you."
            )
        return None
    
    def _get_emotion_context(self) -> str | None:
        """Get emotion-related context from Layer 3 (Emotion).
        
        Returns:
            Emotion context string or None.
        """
        emotion = self.layer_outputs.get("dominant_emotion", "neutral")
        intensity = self.layer_outputs.get("emotion_intensity", 0.5)
        valence = self.layer_outputs.get("valence", 0.0)
        arousal = self.layer_outputs.get("arousal", 0.5)
        
        if emotion == "neutral":
            return None
        
        emotion_descriptions = {
            "excitement": "excited and energized",
            "fear": "fearful and anxious",
            "anger": "frustrated and angry",
            "joy": "happy and optimistic",
            "sadness": "disappointed and down",
            "surprise": "surprised by recent events",
            "disgust": "disgusted by what you're seeing",
            "anticipation": "full of anticipation",
        }
        
        description = emotion_descriptions.get(emotion, emotion)
        
        context = f"- Dominant emotion: You feel {description}"
        
        if intensity > 0.7:
            context += f" (very intense, level: {intensity:.1f})"
        elif intensity > 0.4:
            context += f" (moderate intensity, level: {intensity:.1f})"
        
        if arousal > self.AROUSAL_HIGH_THRESHOLD:
            context += ". Your energy is high and you feel ready to act."
        
        return context
    
    def _get_social_context(self) -> str | None:
        """Get social-related context from Layer 4 (Social Interaction).
        
        Returns:
            Social context string or None.
        """
        social_pressure = self.layer_outputs.get("social_pressure", 0.0)
        herding = self.layer_outputs.get("herding_detected", False)
        viral = self.layer_outputs.get("viral_exposure", False)
        
        parts = []
        
        if social_pressure > self.SOCIAL_PRESSURE_THRESHOLD:
            parts.append(
                f"You feel significant social pressure from the community (level: {social_pressure:.1f})."
            )
        
        if herding:
            parts.append(
                "You notice everyone around you taking similar actions - "
                "the herd is moving in one direction."
            )
        
        if viral:
            parts.append(
                "You've seen viral posts that are energizing the entire community. "
                "The excitement is contagious."
            )
        
        if parts:
            return "- " + " ".join(parts)
        return None
    
    def _get_identity_context(self) -> str | None:
        """Get identity-related context from Layer 5 (Collective Identity).
        
        Returns:
            Identity context string or None.
        """
        identity_state = self.agent_state.get("identity_state")
        if not identity_state:
            return None
        
        group = identity_state.get("primary_group", "NEUTRAL")
        identification = identity_state.get("group_identification", 0.0)
        
        if identification > self.GROUP_IDENTIFICATION_THRESHOLD:
            group_descriptions = {
                "WSB_APE": "the WSB ape community - diamond hands, to the moon!",
                "INSTITUTIONAL": "institutional investors - analytical and measured",
                "RETAIL": "retail investors - cautious but hopeful",
                "CONTRARIAN": "contrarians - going against the crowd",
            }
            description = group_descriptions.get(group, group)
            return (
                f"- You strongly identify with {description} "
                f"(identification strength: {identification:.1f}). "
                "Your group's values guide your decisions."
            )
        return None
    
    def _get_cognitive_context(self) -> str | None:
        """Get cognitive bias context from Layer 2 (Cognition).
        
        Returns:
            Cognitive context string or None.
        """
        social_proof = self.layer_outputs.get("social_proof", 0.0)
        confirmation_bias = self.layer_outputs.get("confirmation_bias", 0.0)
        loss_aversion = self.layer_outputs.get("loss_aversion", 0.0)
        
        parts = []
        
        if social_proof > 0.6:
            parts.append(
                f"You are strongly influenced by social proof (level: {social_proof:.1f}) - "
                "if others are doing it, it must be right."
            )
        
        if confirmation_bias > 0.6:
            parts.append(
                f"You tend to seek information that confirms your existing beliefs "
                f"(confirmation bias: {confirmation_bias:.1f})."
            )
        
        if loss_aversion > 0.7:
            parts.append(
                f"You are highly loss-averse (level: {loss_aversion:.1f}) - "
                "the fear of losing is stronger than the hope of gaining."
            )
        
        if parts:
            return "- " + " ".join(parts)
        return None
    
    def _build_action_guidance(self) -> str:
        """Build action guidance based on psychological state.
        
        Provides subtle guidance to the LLM based on the agent's state.
        
        Returns:
            Action guidance string.
        """
        fomo = self.layer_outputs.get("fomo_level", 0.0)
        stress = self.layer_outputs.get("stress_level", 0.0)
        arousal = self.layer_outputs.get("arousal", 0.5)
        
        if fomo > self.FOMO_HIGH_THRESHOLD and arousal > self.AROUSAL_HIGH_THRESHOLD:
            return (
                "ACTION TENDENCY: Your psychological state suggests a strong urge to engage. "
                "Consider whether to tweet your thoughts or observe more."
            )
        elif stress > self.STRESS_HIGH_THRESHOLD:
            return (
                "ACTION TENDENCY: Your high stress suggests caution. "
                "Consider whether to hold back or seek clarity."
            )
        else:
            return (
                "ACTION TENDENCY: Your state is relatively balanced. "
                "Make a decision that fits your character."
            )
    
    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of the current state for debugging.
        
        Returns:
            Dictionary summarizing key state values.
        """
        return {
            "fomo_level": self.layer_outputs.get("fomo_level", 0.0),
            "stress_level": self.layer_outputs.get("stress_level", 0.0),
            "dominant_emotion": self.layer_outputs.get("dominant_emotion", "neutral"),
            "emotion_intensity": self.layer_outputs.get("emotion_intensity", 0.5),
            "social_pressure": self.layer_outputs.get("social_pressure", 0.0),
            "herding_detected": self.layer_outputs.get("herding_detected", False),
            "viral_exposure": self.layer_outputs.get("viral_exposure", False),
        }
