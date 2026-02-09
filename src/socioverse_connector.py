"""SocioVerse Connector module.

Fetches real user personas from the SocioVerse dataset on HuggingFace.
Reference: https://huggingface.co/datasets/Lishi0905/SocioVerse
"""

import json
import logging
import os
import random
import re
from ast import literal_eval
from typing import Any

from .interfaces import UserPoolProviderABC

logger = logging.getLogger(__name__)
compliance_logger = logging.getLogger("kalsim.compliance")


class SocioVerseConnector(UserPoolProviderABC):
    """Fetches user personas from HuggingFace SocioVerse dataset.
    
    The SocioVerse dataset contains 10M+ real-world user profiles with
    demographic annotations for social simulation research.
    
    Attributes:
        hf_token: HuggingFace authentication token.
    """

    DATASET_NAME = "Lishi0905/SocioVerse"
    DATA_FILE = "user_pool_X.json"
    TERMS_URL = "https://huggingface.co/datasets/Lishi0905/SocioVerse"
    RESEARCH_MODE_ENV = "KALSIM_RESEARCH_MODE"
    _TRUTHY = {"1", "true", "yes", "on"}

    def __init__(
        self,
        hf_token: str | None = None,
        research_mode: bool | None = None,
    ) -> None:
        """Initialize the SocioVerseConnector.
        
        Args:
            hf_token: HuggingFace token. If None, reads from HF_TOKEN env var.
            research_mode:
                If provided, explicitly controls research-only access guard.
                If None, reads from KALSIM_RESEARCH_MODE environment variable.
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.research_mode = (
            research_mode if research_mode is not None else self._read_research_mode_flag()
        )
        self._dataset = None

    def fetch_user_pool(self, count: int = 100) -> list[dict[str, Any]]:
        """Fetch diverse user personas from SocioVerse.
        
        Args:
            count: Number of users to fetch.
            
        Returns:
            List of persona dictionaries in simulation-compatible format.
        """
        if not self.research_mode:
            logger.warning(
                "SocioVerse access blocked: %s must be enabled for research-only use.",
                self.RESEARCH_MODE_ENV,
            )
            self._audit(
                "socioverse_access_blocked",
                reason="research_mode_disabled",
                requested_count=count,
            )
            return []

        try:
            from datasets import load_dataset
            
            logger.info(f"Fetching {count} users from SocioVerse dataset...")
            self._audit("socioverse_access_attempt", requested_count=count)

            dataset = self._load_dataset_with_fallback(load_dataset, count=count)
            
            self._dataset = dataset
            
            personas = [
                self._transform_user(i, dict(user))
                for i, user in enumerate(dataset)
            ]
            
            logger.info(f"Successfully fetched {len(personas)} users from SocioVerse")
            self._audit("socioverse_access_success", loaded_count=len(personas))
            logger.info(
                "SocioVerse compliance note: use this dataset for research experiments only. "
                "For X data, text content must be retrieved via the official X API."
            )
            return personas

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            self._audit("socioverse_access_failed", error_type="ImportError")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch from SocioVerse: {e}")
            hint = self._build_access_hint(e)
            if hint:
                logger.error(hint)
            self._audit(
                "socioverse_access_failed",
                error_type=type(e).__name__,
            )
            return []

    def _load_dataset_with_fallback(self, load_dataset: Any, count: int) -> Any:
        """Load SocioVerse with multiple config fallbacks for cache compatibility."""
        split = f"train[:{count}]"

        # Preferred path: explicit data file.
        try:
            dataset = load_dataset(
                self.DATASET_NAME,
                data_files=self.DATA_FILE,
                split=split,
                token=self.hf_token,
            )
            self._audit(
                "socioverse_dataset_load",
                source="data_files",
                config=f"default-data_files={self.DATA_FILE}",
            )
            return dataset
        except Exception as data_file_error:
            logger.warning(
                "SocioVerse load with data file '%s' failed, trying default config: %s",
                self.DATA_FILE,
                data_file_error,
            )

            # Fallback path: default config often maps to local cache keys.
            try:
                dataset = load_dataset(
                    self.DATASET_NAME,
                    split=split,
                    token=self.hf_token,
                )
                self._audit(
                    "socioverse_dataset_load",
                    source="default",
                )
                return dataset
            except Exception as default_error:
                logger.warning(
                    "SocioVerse default config load failed, trying cached configs: %s",
                    default_error,
                )
                cached_configs = self._extract_cached_configs(
                    str(data_file_error)
                ) or self._extract_cached_configs(str(default_error))
                for config_name in cached_configs:
                    try:
                        logger.info(
                            "Trying cached SocioVerse config: %s",
                            config_name,
                        )
                        dataset = load_dataset(
                            self.DATASET_NAME,
                            config_name,
                            split=split,
                            token=self.hf_token,
                        )
                        self._audit(
                            "socioverse_dataset_load",
                            source="cached_config",
                            config=config_name,
                        )
                        return dataset
                    except Exception as cached_error:
                        logger.debug(
                            "Cached config '%s' failed: %s",
                            config_name,
                            cached_error,
                        )

                # Surface the most relevant failure.
                raise default_error from data_file_error

    @staticmethod
    def _extract_cached_configs(message: str) -> list[str]:
        """Extract cached dataset config names from datasets error messages."""
        if "Available configs in the cache" not in message:
            return []

        match = re.search(
            r"Available configs in the cache:\s*(\[[^\]]*\])",
            message,
        )
        if not match:
            return []

        raw_list = match.group(1)
        try:
            parsed = literal_eval(raw_list)
        except (ValueError, SyntaxError):
            return []

        if not isinstance(parsed, list):
            return []

        return [str(item) for item in parsed if isinstance(item, str)]

    @staticmethod
    def _build_access_hint(error: Exception) -> str | None:
        """Return a targeted guidance message for known access failures."""
        message = str(error).lower()

        if any(token in message for token in ["401", "403", "access denied", "gated"]):
            return (
                f"Ensure you accepted the dataset terms at: {SocioVerseConnector.TERMS_URL}"
            )

        if "couldn't be found on the hugging face hub" in message:
            return (
                "SocioVerse could not be found on Hugging Face Hub. "
                "The connector attempted cached fallbacks."
            )

        return None

    @classmethod
    def _read_research_mode_flag(cls) -> bool:
        """Read research mode guard from environment."""
        raw = os.environ.get(cls.RESEARCH_MODE_ENV, "")
        return raw.strip().lower() in cls._TRUTHY

    def _audit(self, event: str, **fields: Any) -> None:
        """Emit compliance audit logs for SocioVerse access."""
        payload = {
            "event": event,
            "dataset": self.DATASET_NAME,
            "research_mode": self.research_mode,
            **fields,
        }
        compliance_logger.info("socioverse_audit %s", json.dumps(payload, sort_keys=True))

    def _transform_user(self, idx: int, user: dict[str, Any]) -> dict[str, Any]:
        """Transform SocioVerse user schema to simulation persona schema.
        
        Args:
            idx: Index/ID for the user.
            user: Raw user data from SocioVerse dataset.
            
        Returns:
            Persona dictionary compatible with simulation Agent class.
        """
        user_id_raw = user.get("user_id")
        user_id = str(user_id_raw).strip() if user_id_raw is not None else ""
        if not user_id:
            user_id = f"SV_User_{idx}"
        
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
        consumption_raw = user.get("Level of Consumption", "")
        consumption = str(consumption_raw).lower().strip()

        # Keep deterministic behavior when consumption is missing/unknown.
        if not consumption:
            return "neutral"
        
        if "high" in consumption:
            return "bullish"
        elif "low" in consumption:
            return "bearish"
        else:
            # Add randomness to avoid too many neutrals
            # 35% Bullish, 35% Bearish, 30% Neutral
            return random.choices(
                ["bullish", "bearish", "neutral"],
                weights=[0.35, 0.35, 0.30],
                k=1
            )[0]

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
        if not self.research_mode:
            return False

        try:
            from datasets import load_dataset

            dataset = self._load_dataset_with_fallback(load_dataset, count=1)
            return len(dataset) > 0
        except Exception:
            return False
