"""Tests for UserEngine module."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.user_engine import UserEngine, DemographicLabel


class TestDemographicLabel:
    """Tests for DemographicLabel enum."""

    def test_demographic_labels_exist(self):
        """Test that required demographic labels exist."""
        assert DemographicLabel.RETAIL_INVESTOR
        assert DemographicLabel.DAY_TRADER
        assert DemographicLabel.MEME_TRADER
        assert DemographicLabel.INSTITUTIONAL


class TestUserEngine:
    """Tests for UserEngine class."""

    def test_engine_initialization(self):
        """Test initializing UserEngine."""
        engine = UserEngine()
        assert engine is not None

    def test_load_personas_from_list(self, sample_personas):
        """Test loading personas from a provided list."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        assert len(engine.personas) == 3
        assert engine.personas[0]["name"] == "DiamondHandsDave"

    def test_load_personas_from_file(self, sample_personas, tmp_path):
        """Test loading personas from a JSON file."""
        persona_file = tmp_path / "test_personas.json"
        with open(persona_file, "w") as f:
            json.dump(sample_personas, f)

        engine = UserEngine()
        engine.load_personas(filepath=persona_file)

        assert len(engine.personas) == 3

    def test_assign_demographic_label_high_risk(self, sample_persona):
        """Test assigning demographic label to high-risk persona."""
        engine = UserEngine()
        engine.load_personas(personas=[sample_persona])

        label = engine.get_demographic_label(0)
        assert label in [DemographicLabel.MEME_TRADER, DemographicLabel.DAY_TRADER]

    def test_assign_demographic_label_low_risk(self):
        """Test assigning demographic label to low-risk persona."""
        persona = {
            "id": 0,
            "name": "Conservative",
            "personality_traits": ["cautious", "analytical"],
            "beliefs": {"risk_tolerance": "low", "trust_in_institutions": "high"},
        }
        engine = UserEngine()
        engine.load_personas(personas=[persona])

        label = engine.get_demographic_label(0)
        assert label == DemographicLabel.RETAIL_INVESTOR

    def test_get_persona_by_id(self, sample_personas):
        """Test retrieving persona by ID."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        persona = engine.get_persona(1)
        assert persona["name"] == "CautiousCarla"

    def test_get_persona_invalid_id(self, sample_personas):
        """Test retrieving persona with invalid ID."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        with pytest.raises(KeyError):
            engine.get_persona(999)

    def test_get_all_persona_ids(self, sample_personas):
        """Test getting all persona IDs."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        ids = engine.get_all_ids()
        assert len(ids) == 3
        assert 0 in ids
        assert 1 in ids
        assert 2 in ids

    def test_filter_by_demographic(self, sample_personas):
        """Test filtering personas by demographic label."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        high_risk = engine.filter_by_risk_tolerance("high")
        assert len(high_risk) >= 1

    def test_get_influence_score(self, sample_personas):
        """Test getting influence score for a persona."""
        engine = UserEngine()
        engine.load_personas(personas=sample_personas)

        score = engine.get_influence_score(2)
        assert score == 0.85
