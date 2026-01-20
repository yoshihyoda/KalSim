"""User Engine module for persona management.

Handles loading, categorizing, and querying user personas.
"""

import json
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DemographicLabel(Enum):
    """Demographic labels for categorizing users."""
    RETAIL_INVESTOR = auto()
    DAY_TRADER = auto()
    MEME_TRADER = auto()
    INSTITUTIONAL = auto()
    LURKER = auto()


class UserEngine:
    """Manages user personas and their demographic classification.
    
    Provides methods for loading personas from files or lists,
    assigning demographic labels, and querying persona attributes.
    
    Attributes:
        personas: Dictionary mapping persona ID to persona data.
        demographic_labels: Dictionary mapping persona ID to DemographicLabel.
    """

    def __init__(self) -> None:
        """Initialize the UserEngine."""
        self.personas: dict[int, dict[str, Any]] = {}
        self.demographic_labels: dict[int, DemographicLabel] = {}

    def load_personas(
        self,
        filepath: Path | None = None,
        personas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Load personas from a file or provided list.
        
        Args:
            filepath: Path to JSON file containing personas.
            personas: List of persona dictionaries.
            
        Raises:
            ValueError: If neither filepath nor personas is provided.
        """
        if personas is not None:
            persona_list = personas
        elif filepath is not None:
            with open(filepath, "r", encoding="utf-8") as f:
                persona_list = json.load(f)
        else:
            raise ValueError("Either filepath or personas must be provided")

        for persona in persona_list:
            persona_id = persona.get("id", len(self.personas))
            self.personas[persona_id] = persona
            self.demographic_labels[persona_id] = self._assign_demographic(persona)

        logger.info(f"Loaded {len(self.personas)} personas")

    def _assign_demographic(self, persona: dict[str, Any]) -> DemographicLabel:
        """Assign a demographic label based on persona traits.
        
        Args:
            persona: Persona dictionary.
            
        Returns:
            Appropriate DemographicLabel.
        """
        beliefs = persona.get("beliefs", {})
        traits = persona.get("personality_traits", [])
        interests = persona.get("interests", [])

        risk_tolerance = beliefs.get("risk_tolerance", "moderate")
        trust_institutions = beliefs.get("trust_in_institutions", "moderate")

        trait_set = set(t.lower() for t in traits)
        interest_set = set(i.lower() for i in interests)

        meme_indicators = {"wsb", "memes", "reddit", "crypto"}
        if interest_set & meme_indicators and risk_tolerance == "high":
            return DemographicLabel.MEME_TRADER

        if risk_tolerance == "high" and "impulsive" in trait_set:
            return DemographicLabel.DAY_TRADER

        if trust_institutions == "high" and risk_tolerance == "low":
            return DemographicLabel.RETAIL_INVESTOR

        if "analytical" in trait_set and "quantitative finance" in interest_set:
            return DemographicLabel.INSTITUTIONAL

        if "lurking" in trait_set or persona.get("social", {}).get("influence_score", 0.5) < 0.2:
            return DemographicLabel.LURKER

        return DemographicLabel.RETAIL_INVESTOR

    def get_demographic_label(self, persona_id: int) -> DemographicLabel:
        """Get the demographic label for a persona.
        
        Args:
            persona_id: ID of the persona.
            
        Returns:
            DemographicLabel for the persona.
        """
        return self.demographic_labels.get(persona_id, DemographicLabel.RETAIL_INVESTOR)

    def get_persona(self, persona_id: int) -> dict[str, Any]:
        """Get a persona by ID.
        
        Args:
            persona_id: ID of the persona.
            
        Returns:
            Persona dictionary.
            
        Raises:
            KeyError: If persona ID not found.
        """
        if persona_id not in self.personas:
            raise KeyError(f"Persona {persona_id} not found")
        return self.personas[persona_id]

    def get_all_ids(self) -> list[int]:
        """Get all persona IDs.
        
        Returns:
            List of all persona IDs.
        """
        return list(self.personas.keys())

    def filter_by_risk_tolerance(self, tolerance: str) -> list[dict[str, Any]]:
        """Filter personas by risk tolerance.
        
        Args:
            tolerance: Risk tolerance level ('low', 'moderate', 'high').
            
        Returns:
            List of matching personas.
        """
        return [
            p for p in self.personas.values()
            if p.get("beliefs", {}).get("risk_tolerance") == tolerance
        ]

    def filter_by_demographic(self, label: DemographicLabel) -> list[dict[str, Any]]:
        """Filter personas by demographic label.
        
        Args:
            label: DemographicLabel to filter by.
            
        Returns:
            List of matching personas.
        """
        matching_ids = [
            pid for pid, lbl in self.demographic_labels.items()
            if lbl == label
        ]
        return [self.personas[pid] for pid in matching_ids]

    def get_influence_score(self, persona_id: int) -> float:
        """Get the influence score for a persona.
        
        Args:
            persona_id: ID of the persona.
            
        Returns:
            Influence score (0.0 to 1.0).
        """
        persona = self.get_persona(persona_id)
        return persona.get("social", {}).get("influence_score", 0.5)

    def get_follower_count(self, persona_id: int) -> int:
        """Get the follower count for a persona.
        
        Args:
            persona_id: ID of the persona.
            
        Returns:
            Number of followers.
        """
        persona = self.get_persona(persona_id)
        return persona.get("social", {}).get("follower_count", 0)
