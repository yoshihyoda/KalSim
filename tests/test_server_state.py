"""Tests for server state endpoint calculations."""

import pytest
from pydantic import ValidationError

from src import server
from src.config import STEPS_PER_DAY


class DummySimulation:
    """Minimal simulation-like object for server state tests."""

    def __init__(self, days: int, current_step: int):
        self.days = days
        self._current_step_index = current_step
        self._current_price = 61.5
        self.simulation_log = [
            {
                "agent_id": 1,
                "timestamp": "2026-01-01T00:00:00",
                "action_type": "TWEET",
                "content": "hello",
                "metadata": {},
            }
        ]


def test_get_state_uses_steps_per_day_constant(monkeypatch):
    """State endpoint should compute totals and day using STEPS_PER_DAY."""
    local_state = server.SimulationState()
    local_state.simulation = DummySimulation(days=2, current_step=STEPS_PER_DAY + 3)
    local_state.is_running = True

    monkeypatch.setattr(server, "state", local_state)

    status = server.get_state()

    assert status.total_steps == 2 * STEPS_PER_DAY
    assert status.current_day == 2
    assert status.current_step == STEPS_PER_DAY + 3


def test_get_state_without_simulation_returns_defaults(monkeypatch):
    """State endpoint should return safe defaults when simulation is absent."""
    local_state = server.SimulationState()
    local_state.error = "boom"

    monkeypatch.setattr(server, "state", local_state)

    status = server.get_state()

    assert status.is_running is False
    assert status.current_step == 0
    assert status.total_steps == 0
    assert status.current_day == 0
    assert status.run_error == "boom"


def test_validate_agent_personas_coerces_numeric_name():
    """Agent payloads with numeric names should not fail validation."""
    raw_agents = [
        {
            "id": 0,
            "name": 237236420,
            "personality_traits": ["analytical"],
            "interests": ["markets"],
            "beliefs": {"risk_tolerance": "moderate"},
            "social": {"follower_count": 100, "influence_score": 0.1},
        }
    ]

    personas = server._validate_agent_personas(raw_agents)

    assert len(personas) == 1
    assert personas[0].name == "237236420"


def test_validate_agent_personas_fills_blank_name():
    """Blank names should be replaced with a deterministic fallback."""
    raw_agents = [{"id": 1, "name": "   "}]

    personas = server._validate_agent_personas(raw_agents)

    assert personas[0].name == "Agent_0"


def test_validate_agent_personas_drops_restricted_content_fields():
    """Agent payloads should not include raw social text fields."""
    raw_agents = [
        {
            "id": 1,
            "name": "AgentX",
            "tweet_text": "raw content",
            "posts": ["x", "y"],
            "personality_traits": ["calm"],
        }
    ]

    personas = server._validate_agent_personas(raw_agents)
    payload = personas[0].model_dump(exclude_none=True)

    assert "tweet_text" not in payload
    assert "posts" not in payload
    assert payload["name"] == "AgentX"


def test_kalshi_agents_request_validates_count_bounds():
    """Agent generation count should be constrained to safe bounds."""
    with pytest.raises(ValidationError):
        server.KalshiAgentsRequest(event_ticker="TEST", count=0)

    with pytest.raises(ValidationError):
        server.KalshiAgentsRequest(event_ticker="TEST", count=101)
