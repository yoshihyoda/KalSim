"""Layer 4: Social Interaction Module.

Models emotion contagion, social influence, and herding behavior.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmotionContagion:
    """Models emotional contagion between connected agents.
    
    Emotions spread through social network connections with
    strength modulated by connection weight and susceptibility.
    
    Attributes:
        susceptibility: Base susceptibility to emotion contagion (0.0-1.0).
    """

    def __init__(self, susceptibility: float = 0.5) -> None:
        """Initialize EmotionContagion.
        
        Args:
            susceptibility: Base susceptibility to contagion.
        """
        self.susceptibility = susceptibility

    def propagate(
        self,
        source_emotion: dict[str, float],
        target_emotion: dict[str, float],
        connection_strength: float,
    ) -> dict[str, float]:
        """Propagate emotion from source to target.
        
        Args:
            source_emotion: Source agent's emotion (valence, arousal).
            target_emotion: Target agent's current emotion.
            connection_strength: Strength of connection (0.0-1.0).
            
        Returns:
            Updated target emotion after contagion.
        """
        influence_factor = self.susceptibility * connection_strength

        new_valence = (
            target_emotion["valence"] * (1 - influence_factor)
            + source_emotion["valence"] * influence_factor
        )

        new_arousal = (
            target_emotion["arousal"] * (1 - influence_factor)
            + source_emotion["arousal"] * influence_factor
        )

        return {
            "valence": max(-1.0, min(1.0, new_valence)),
            "arousal": max(0.0, min(1.0, new_arousal)),
        }


class SocialInfluence:
    """Models social influence between agents.
    
    Calculates influence based on relative status, follower counts,
    and other social metrics.
    """

    def __init__(self) -> None:
        """Initialize SocialInfluence."""
        pass

    def calculate(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> float:
        """Calculate influence of source on target.
        
        Args:
            source: Source agent attributes.
            target: Target agent attributes.
            
        Returns:
            Influence score (0.0-1.0).
        """
        source_influence = source.get("influence_score", 0.5)
        target_influence = target.get("influence_score", 0.5)

        source_followers = source.get("follower_count", 100)
        target_followers = target.get("follower_count", 100)

        if target_followers == 0:
            follower_ratio = 1.0
        else:
            follower_ratio = min(source_followers / max(target_followers, 1), 10.0) / 10.0

        influence_gap = max(0, source_influence - target_influence)

        influence = 0.3 * source_influence + 0.3 * follower_ratio + 0.4 * influence_gap
        return min(1.0, influence)


class SocialInteractionModule:
    """Models social interactions and their effects on agents.
    
    Handles emotion contagion, social influence, and herding behavior.
    
    Attributes:
        name: Module identifier.
        contagion: EmotionContagion instance.
        influence: SocialInfluence instance.
    """

    def __init__(self, susceptibility: float = 0.5) -> None:
        """Initialize the SocialInteractionModule.
        
        Args:
            susceptibility: Susceptibility to emotion contagion.
        """
        self.name = "social_interaction"
        self.contagion = EmotionContagion(susceptibility=susceptibility)
        self.influence = SocialInfluence()
        self._last_social_pressure = 0.0
        self._last_emotion_received: dict[str, float] | None = None
        self._herding_threshold = 0.7

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process social interaction state.
        
        Calculates emotion contagion, social pressure, and herding.
        
        Args:
            state: Combined state including agent and social data.
            
        Returns:
            Dictionary with social interaction outputs.
        """
        agent = state.get("agent", {})
        social = state.get("social", {})

        neighbors = social.get("neighbors", [])
        connection_strengths = social.get("connection_strengths", {})

        agent_emotion = agent.get("emotion", {"valence": 0.0, "arousal": 0.5})

        if neighbors:
            weights = [
                connection_strengths.get(n.get("id", i), 0.5)
                for i, n in enumerate(neighbors)
            ]
            aggregated = self.aggregate_emotions(neighbors, weights)

            avg_connection = sum(weights) / len(weights) if weights else 0.5
            emotion_received = self.contagion.propagate(
                aggregated, agent_emotion, avg_connection
            )
        else:
            emotion_received = agent_emotion.copy()

        self._last_emotion_received = emotion_received

        consensus_strength = social.get("consensus_strength", 0.0)
        neighbor_count = len(neighbors)
        social_pressure = consensus_strength * min(neighbor_count / 10, 1.0)
        self._last_social_pressure = social_pressure

        herding_detected = self._detect_herding(neighbors)

        return {
            "emotion_received": emotion_received,
            "social_pressure": social_pressure,
            "herding_detected": herding_detected,
            "neighbor_influence": neighbor_count / 10 if neighbors else 0.0,
            "emotional_alignment": self._calculate_alignment(
                agent_emotion, emotion_received
            ),
        }

    def aggregate_emotions(
        self,
        neighbors: list[dict[str, Any]],
        weights: list[float],
    ) -> dict[str, float]:
        """Aggregate emotions from multiple neighbors.
        
        Args:
            neighbors: List of neighbor data with emotions.
            weights: Connection weights for each neighbor.
            
        Returns:
            Weighted average emotion.
        """
        if not neighbors:
            return {"valence": 0.0, "arousal": 0.5}

        total_weight = sum(weights) or 1.0
        normalized_weights = [w / total_weight for w in weights]

        valence_sum = 0.0
        arousal_sum = 0.0

        for neighbor, weight in zip(neighbors, normalized_weights):
            emotion = neighbor.get("emotion", {"valence": 0.0, "arousal": 0.5})
            valence_sum += emotion.get("valence", 0.0) * weight
            arousal_sum += emotion.get("arousal", 0.5) * weight

        return {"valence": valence_sum, "arousal": arousal_sum}

    def _detect_herding(self, neighbors: list[dict[str, Any]]) -> bool:
        """Detect herding behavior among neighbors.
        
        Args:
            neighbors: List of neighbor data.
            
        Returns:
            True if herding is detected.
        """
        if len(neighbors) < 3:
            return False

        actions = [n.get("action") for n in neighbors if n.get("action")]
        if not actions:
            return False

        action_counts: dict[str, int] = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        max_count = max(action_counts.values())
        return (max_count / len(actions)) >= self._herding_threshold

    def _calculate_alignment(
        self,
        original: dict[str, float],
        received: dict[str, float],
    ) -> float:
        """Calculate emotional alignment between original and received.
        
        Args:
            original: Original emotion state.
            received: Emotion after contagion.
            
        Returns:
            Alignment score (0.0-1.0).
        """
        valence_diff = abs(original.get("valence", 0) - received.get("valence", 0))
        arousal_diff = abs(original.get("arousal", 0.5) - received.get("arousal", 0.5))

        return 1.0 - (valence_diff + arousal_diff) / 2

    def get_state_summary(self) -> str:
        """Get a summary of current social interaction state.
        
        Returns:
            Human-readable state summary.
        """
        emotion_str = "None"
        if self._last_emotion_received:
            emotion_str = (
                f"v={self._last_emotion_received.get('valence', 0):.2f}, "
                f"a={self._last_emotion_received.get('arousal', 0):.2f}"
            )

        return (
            f"Social pressure: {self._last_social_pressure:.2f}, "
            f"Emotion received: {emotion_str}"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self._last_social_pressure = 0.0
        self._last_emotion_received = None
