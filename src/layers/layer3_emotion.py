"""Layer 3: Emotion Module.

Models emotional states, decay dynamics, and stimulus response.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    """State of an agent's emotional state.
    
    Uses valence-arousal model of emotion.
    
    Attributes:
        valence: Positive/negative dimension (-1.0 to 1.0).
        arousal: Activation level (0.0 to 1.0).
        dominant_emotion: Label for the dominant emotion.
        intensity: Overall emotional intensity.
    """
    valence: float = 0.0
    arousal: float = 0.5
    dominant_emotion: str = "neutral"
    intensity: float = 0.5


class EmotionModule:
    """Models emotional dynamics and stimulus response.
    
    Handles emotion decay, amplification, and classification.
    
    Attributes:
        name: Module identifier.
        decay_rate: Rate of emotion decay toward neutral.
    """

    EMOTION_MAP = {
        ("positive", "high"): "excitement",
        ("positive", "low"): "contentment",
        ("negative", "high"): "fear",
        ("negative", "low"): "sadness",
        ("neutral", "high"): "alertness",
        ("neutral", "low"): "calm",
    }

    def __init__(self, decay_rate: float = 0.1) -> None:
        """Initialize the EmotionModule.
        
        Args:
            decay_rate: Rate of decay toward neutral (0.0-1.0).
        """
        self.name = "emotion"
        self.decay_rate = decay_rate
        self._current_state = EmotionState()

    def apply_decay(self, state: EmotionState, time_steps: int = 1) -> EmotionState:
        """Apply emotional decay over time.
        
        Emotions decay exponentially toward neutral state.
        
        Args:
            state: Current emotion state.
            time_steps: Number of time steps elapsed.
            
        Returns:
            Decayed EmotionState.
        """
        decay_factor = math.exp(-self.decay_rate * time_steps)

        new_valence = state.valence * decay_factor
        new_arousal = 0.5 + (state.arousal - 0.5) * decay_factor

        new_emotion = self.classify_emotion(new_valence, new_arousal)
        new_intensity = self.calculate_intensity(new_valence, new_arousal)

        return EmotionState(
            valence=new_valence,
            arousal=new_arousal,
            dominant_emotion=new_emotion,
            intensity=new_intensity,
        )

    def amplify(self, state: EmotionState, factor: float) -> EmotionState:
        """Amplify current emotional state.
        
        Args:
            state: Current emotion state.
            factor: Amplification factor (>1 amplifies, <1 dampens).
            
        Returns:
            Amplified EmotionState.
        """
        new_valence = state.valence * factor
        new_valence = max(-1.0, min(1.0, new_valence))

        arousal_delta = (state.arousal - 0.5) * factor
        new_arousal = 0.5 + arousal_delta
        new_arousal = max(0.0, min(1.0, new_arousal))

        new_emotion = self.classify_emotion(new_valence, new_arousal)
        new_intensity = self.calculate_intensity(new_valence, new_arousal)

        return EmotionState(
            valence=new_valence,
            arousal=new_arousal,
            dominant_emotion=new_emotion,
            intensity=new_intensity,
        )

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process emotional state with stimulus.
        
        Args:
            state: Combined state including agent emotion and stimulus.
            
        Returns:
            Dictionary with emotion outputs.
        """
        agent = state.get("agent", {})
        stimulus = state.get("stimulus", {})

        current_emotion = agent.get("emotion", {"valence": 0.0, "arousal": 0.5})
        valence = current_emotion.get("valence", 0.0)
        arousal = current_emotion.get("arousal", 0.5)

        stimulus_type = stimulus.get("type", "")
        intensity = stimulus.get("intensity", 0.0)

        if stimulus_type == "market_surge":
            valence += intensity * 0.5
            arousal += intensity * 0.3
        elif stimulus_type == "market_crash":
            valence -= intensity * 0.6
            arousal += intensity * 0.4
        elif stimulus_type == "viral_post":
            valence += intensity * 0.3
            arousal += intensity * 0.2
        elif stimulus_type == "fud":
            valence -= intensity * 0.4
            arousal += intensity * 0.3

        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        dominant_emotion = self.classify_emotion(valence, arousal)
        emotion_intensity = self.calculate_intensity(valence, arousal)

        self._current_state = EmotionState(
            valence=valence,
            arousal=arousal,
            dominant_emotion=dominant_emotion,
            intensity=emotion_intensity,
        )

        return {
            "valence": valence,
            "arousal": arousal,
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": emotion_intensity,
        }

    def classify_emotion(self, valence: float, arousal: float) -> str:
        """Classify emotion from valence and arousal.
        
        Args:
            valence: Valence value (-1 to 1).
            arousal: Arousal value (0 to 1).
            
        Returns:
            Emotion label string.
        """
        if valence > 0.6 and arousal > 0.7:
            return "euphoria"
        elif valence > 0.3 and arousal > 0.6:
            return "excitement"
        elif valence > 0.3 and arousal < 0.4:
            return "contentment"
        elif valence < -0.6 and arousal > 0.7:
            return "panic"
        elif valence < -0.3 and arousal > 0.6:
            return "fear"
        elif valence < -0.3 and arousal > 0.4:
            return "anxiety"
        elif valence < -0.3 and arousal < 0.4:
            return "sadness"
        elif arousal > 0.6:
            return "alertness"
        elif arousal < 0.3:
            return "calm"
        else:
            return "neutral"

    def calculate_intensity(self, valence: float, arousal: float) -> float:
        """Calculate overall emotional intensity.
        
        Args:
            valence: Valence value.
            arousal: Arousal value.
            
        Returns:
            Intensity value (0.0 to 1.0).
        """
        valence_magnitude = abs(valence)
        arousal_deviation = abs(arousal - 0.5) * 2

        intensity = (valence_magnitude + arousal_deviation) / 2
        return min(1.0, intensity)

    def get_state_summary(self) -> str:
        """Get a summary of current emotion state.
        
        Returns:
            Human-readable state summary.
        """
        return (
            f"Emotion: {self._current_state.dominant_emotion}, "
            f"Valence: {self._current_state.valence:.2f}, "
            f"Arousal: {self._current_state.arousal:.2f}, "
            f"Intensity: {self._current_state.intensity:.2f}"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self._current_state = EmotionState()
