"""Tests for Layer 3: Emotion module."""

import pytest

from src.layers.layer3_emotion import EmotionModule, EmotionState


class TestEmotionState:
    """Tests for EmotionState dataclass."""

    def test_state_creation(self):
        """Test creating EmotionState."""
        state = EmotionState(
            valence=0.5,
            arousal=0.7,
            dominant_emotion="excitement",
        )
        assert state.valence == 0.5
        assert state.arousal == 0.7
        assert state.dominant_emotion == "excitement"

    def test_state_default_values(self):
        """Test default values for EmotionState."""
        state = EmotionState()
        assert state.valence == 0.0
        assert state.arousal == 0.5
        assert state.dominant_emotion == "neutral"


class TestEmotionModule:
    """Tests for EmotionModule class."""

    def test_module_initialization(self):
        """Test initializing EmotionModule."""
        module = EmotionModule()
        assert module.name == "emotion"
        assert module.decay_rate > 0

    def test_decay_emotion_over_time(self):
        """Test that emotions decay over time."""
        module = EmotionModule(decay_rate=0.1)

        initial_state = EmotionState(
            valence=0.8,
            arousal=0.9,
            dominant_emotion="excitement",
        )

        decayed = module.apply_decay(initial_state, time_steps=5)

        assert decayed.valence < initial_state.valence
        assert decayed.arousal < initial_state.arousal

    def test_excitement_decays_toward_neutral(self):
        """Test that excitement decays toward neutral."""
        module = EmotionModule(decay_rate=0.2)

        state = EmotionState(valence=0.9, arousal=0.9, dominant_emotion="excitement")

        for _ in range(10):
            state = module.apply_decay(state, time_steps=1)

        assert abs(state.valence) < 0.5
        assert state.arousal < 0.7

    def test_amplify_emotion(self):
        """Test emotion amplification."""
        module = EmotionModule()

        initial = EmotionState(valence=0.3, arousal=0.6, dominant_emotion="interest")

        amplified = module.amplify(initial, factor=1.5)

        assert amplified.valence > initial.valence
        assert amplified.arousal > initial.arousal

    def test_amplify_respects_bounds(self):
        """Test that amplification respects bounds."""
        module = EmotionModule()

        initial = EmotionState(valence=0.8, arousal=0.9)

        amplified = module.amplify(initial, factor=2.0)

        assert amplified.valence <= 1.0
        assert amplified.arousal <= 1.0

    def test_process_with_stimulus(self):
        """Test processing emotion with stimulus."""
        module = EmotionModule()

        state = {
            "agent": {
                "emotion": {"valence": 0.3, "arousal": 0.4},
            },
            "stimulus": {
                "type": "market_surge",
                "intensity": 0.8,
            },
        }

        result = module.process(state)

        assert "valence" in result
        assert "arousal" in result
        assert "dominant_emotion" in result

    def test_process_market_surge_increases_excitement(self):
        """Test that market surge increases excitement."""
        module = EmotionModule()

        state = {
            "agent": {
                "emotion": {"valence": 0.2, "arousal": 0.3},
            },
            "stimulus": {
                "type": "market_surge",
                "intensity": 0.9,
            },
        }

        result = module.process(state)

        assert result["valence"] > 0.2
        assert result["arousal"] > 0.3

    def test_process_market_crash_increases_fear(self):
        """Test that market crash increases fear."""
        module = EmotionModule()

        state = {
            "agent": {
                "emotion": {"valence": 0.3, "arousal": 0.3},
            },
            "stimulus": {
                "type": "market_crash",
                "intensity": 0.8,
            },
        }

        result = module.process(state)

        assert result["valence"] < 0.3
        assert result["arousal"] > 0.3

    def test_classify_emotion(self):
        """Test emotion classification from valence/arousal."""
        module = EmotionModule()

        excited = module.classify_emotion(0.8, 0.9)
        assert excited in ["excitement", "euphoria"]

        fearful = module.classify_emotion(-0.7, 0.8)
        assert fearful in ["fear", "anxiety", "panic"]

        calm = module.classify_emotion(0.1, 0.2)
        assert calm in ["calm", "neutral", "contentment"]

    def test_emotion_intensity(self):
        """Test calculating emotion intensity."""
        module = EmotionModule()

        high_intensity = module.calculate_intensity(0.9, 0.9)
        low_intensity = module.calculate_intensity(0.1, 0.2)

        assert high_intensity > low_intensity
        assert 0 <= high_intensity <= 1
        assert 0 <= low_intensity <= 1

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = EmotionModule()
        module._current_state = EmotionState(
            valence=0.7, arousal=0.8, dominant_emotion="excitement"
        )

        summary = module.get_state_summary()

        assert "excitement" in summary.lower() or "0.7" in summary

    def test_reset(self):
        """Test resetting module state."""
        module = EmotionModule()
        module._current_state = EmotionState(valence=0.9, arousal=0.9)

        module.reset()

        assert module._current_state.valence == 0.0
        assert module._current_state.dominant_emotion == "neutral"
