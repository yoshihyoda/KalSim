"""SocioVerse Connector module.

Fetches real user personas from the SocioVerse dataset on HuggingFace.
Reference: https://huggingface.co/datasets/Lishi0905/SocioVerse
"""

import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)


class SocioVerseConnector:
    """Fetches user personas from HuggingFace SocioVerse dataset.
    
    The SocioVerse dataset contains 10M+ real-world user profiles with
    demographic annotations for social simulation research.
    
    Attributes:
        hf_token: HuggingFace authentication token.
    """

    DATASET_NAME = "Lishi0905/SocioVerse"
    DATA_FILE = "user_pool_X.json"

    def __init__(self, hf_token: str | None = None) -> None:
        """Initialize the SocioVerseConnector.
        
        Args:
            hf_token: HuggingFace token. If None, reads from HF_TOKEN env var.
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._dataset = None

    def fetch_user_pool(self, count: int = 100) -> list[dict[str, Any]]:
        """Fetch diverse user personas from SocioVerse.
        
        Args:
            count: Number of users to fetch.
            
        Returns:
            List of persona dictionaries in simulation-compatible format.
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"Fetching {count} users from SocioVerse dataset...")
            
            dataset = load_dataset(
                self.DATASET_NAME,
                data_files=self.DATA_FILE,
                split=f"train[:{count}]",
                token=self.hf_token,
            )
            
            self._dataset = dataset
            
            personas = [
                self._transform_user(i, dict(user))
                for i, user in enumerate(dataset)
            ]
            
            logger.info(f"Successfully fetched {len(personas)} users from SocioVerse")
            return personas

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch from SocioVerse: {e}")
            logger.error(
                "Ensure you have accepted the dataset terms at: "
                "https://huggingface.co/datasets/Lishi0905/SocioVerse"
            )
            return []

    def _transform_user(self, idx: int, user: dict[str, Any]) -> dict[str, Any]:
        """Transform SocioVerse user schema to simulation persona schema.
        
        Args:
            idx: Index/ID for the user.
            user: Raw user data from SocioVerse dataset.
            
        Returns:
            Persona dictionary compatible with simulation Agent class.
        """
        user_id = user.get("user_id", f"SV_User_{idx}")
        
        personality_traits = self._extract_traits(user)
        interests = self._extract_interests(user)
        risk_tolerance = self._map_consumption_to_risk(user)
        influence = user.get("influence", 0.5)
        
        return {
            "id": idx,
            "name": user_id,
            "personality_traits": personality_traits,
            "interests": interests,
            "beliefs": {
                "risk_tolerance": risk_tolerance,
                "market_outlook": self._infer_market_outlook(user),
                "trust_in_institutions": self._infer_trust_level(user),
            },
            "social": {
                "follower_count": int(influence * 10000),
                "influence_score": float(influence),
            },
            "demographics": {
                "age": user.get("AGE", "unknown"),
                "gender": user.get("GENDER", "unknown"),
                "education": user.get("Education", "unknown"),
            },
        }

    def _extract_traits(self, user: dict[str, Any]) -> list[str]:
        """Extract personality traits from user data.
        
        Args:
            user: Raw user data.
            
        Returns:
            List of personality trait strings.
        """
        traits = []
        
        age = user.get("AGE", "")
        if age:
            if "18" in str(age) or "25" in str(age):
                traits.append("young")
            elif "45" in str(age) or "55" in str(age):
                traits.append("experienced")
        
        consumption = user.get("Level of Consumption", "")
        if consumption:
            consumption_lower = str(consumption).lower()
            if "high" in consumption_lower:
                traits.append("risk-seeking")
            elif "low" in consumption_lower:
                traits.append("cautious")
            else:
                traits.append("moderate")
        
        education = user.get("Education", "")
        if education:
            edu_lower = str(education).lower()
            if "graduate" in edu_lower or "master" in edu_lower or "phd" in edu_lower:
                traits.append("analytical")
            elif "high school" in edu_lower:
                traits.append("practical")
        
        if len(traits) < 2:
            default_traits = ["curious", "social", "engaged", "observant"]
            traits.extend(random.sample(default_traits, 2 - len(traits)))
        
        return traits[:3]

    def _extract_interests(self, user: dict[str, Any]) -> list[str]:
        """Extract interests from user data.
        
        Args:
            user: Raw user data.
            
        Returns:
            List of interest strings.
        """
        interests = ["prediction markets", "news", "finance"]
        
        age = user.get("AGE", "")
        if age and ("18" in str(age) or "25" in str(age)):
            interests.extend(["social media", "technology"])
        elif age and ("45" in str(age) or "55" in str(age)):
            interests.extend(["investing", "politics"])
        
        return interests[:4]

    def _map_consumption_to_risk(self, user: dict[str, Any]) -> str:
        """Map consumption level to risk tolerance.
        
        Args:
            user: Raw user data.
            
        Returns:
            Risk tolerance string: 'high', 'moderate', or 'low'.
        """
        consumption = str(user.get("Level of Consumption", "")).lower()
        
        if "high" in consumption:
            return "high"
        elif "low" in consumption:
            return "low"
        else:
            return "moderate"

    def _infer_market_outlook(self, user: dict[str, Any]) -> str:
        """Infer market outlook from user attributes.
        
        Args:
            user: Raw user data.
            
        Returns:
            Market outlook: 'bullish', 'bearish', or 'neutral'.
        """
        consumption = str(user.get("Level of Consumption", "")).lower()
        
        if "high" in consumption:
            return "bullish"
        elif "low" in consumption:
            return "bearish"
        else:
            return "neutral"

    def _infer_trust_level(self, user: dict[str, Any]) -> str:
        """Infer institutional trust level from user attributes.
        
        Args:
            user: Raw user data.
            
        Returns:
            Trust level: 'high', 'moderate', or 'low'.
        """
        education = str(user.get("Education", "")).lower()
        
        if "graduate" in education or "master" in education:
            return "moderate"
        elif "high school" in education:
            return "low"
        else:
            return "moderate"

    def is_available(self) -> bool:
        """Check if SocioVerse dataset is accessible.
        
        Returns:
            True if dataset can be accessed, False otherwise.
        """
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                self.DATASET_NAME,
                data_files=self.DATA_FILE,
                split="train[:1]",
                token=self.hf_token,
            )
            return len(dataset) > 0
        except Exception:
            return False
