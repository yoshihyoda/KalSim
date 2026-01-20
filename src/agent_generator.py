"""Agent Generator module.

Uses LLM to generate diverse agent personas based on specific market trends.
"""

import json
import logging
import re
from typing import Any

from .llm_interface import LlamaInterface

logger = logging.getLogger(__name__)


class AgentGenerator:
    """Generates agent personas using LLM."""
    
    def __init__(self, llm: LlamaInterface):
        self.llm = llm
    
    def generate_agents(self, trend_summary: str, count: int = 5) -> list[dict[str, Any]]:
        """Generate agent personas based on trends.
        
        Args:
            trend_summary: Description of current market trends/topics.
            count: Number of agents to generate.
            
        Returns:
            List of agent persona dictionaries.
        """
        prompt = f"""Create {count} diverse and realistic user personas who are interested in these specific market topics:
"{trend_summary}"

For each user, provide a JSON object with:
- "name": A creative username.
- "personality_traits": List of 3 traits (e.g., "Risk-taker", "Analytical", "Meme-lover").
- "interests": List of 3 interests related to the topics.
- "beliefs": A dictionary with "risk_tolerance" ("High", "Moderate", "Low") and "view" ("Bullish" or "Bearish" on the topics).

Output ONLY a valid JSON array of objects. Do not include markdown formatting or extra text.
Example format:
[
  {{
    "name": "CryptoKing",
    "personality_traits": ["Bold", "Impulsive"],
    "interests": ["Bitcoin", "Tech"],
    "beliefs": {{"risk_tolerance": "High", "view": "Bullish"}}
  }}
]
"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.9)
            cleaned_response = self._clean_json_response(response)
            agents = json.loads(cleaned_response)
            
            # Validation: ensure we have a list
            if not isinstance(agents, list):
                logger.error("LLM did not return a list. Wrapped content.")
                agents = [agents]
                
            # ensure valid count (truncate or pad if needed, though usually LLM obeys)
            return agents[:count]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")
            # Fallback
            return self._get_fallback_agents(count)
        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            return self._get_fallback_agents(count)

    def _clean_json_response(self, text: str) -> str:
        """Clean LLM output to ensure valid JSON."""
        # Remove potential markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # If the model added prose around the JSON, extract the first valid block.
        extracted = self._extract_json_block(text)
        return extracted.strip() if extracted else text

    def _extract_json_block(self, text: str) -> str | None:
        """Extract the first valid JSON object/array from a string."""
        if not text:
            return None
        if text[0] in "[{":
            return text

        for opener in ("[", "{"):
            start = text.find(opener)
            if start == -1:
                continue
            end = self._find_matching_bracket(text, start, opener)
            if end is None:
                continue
            candidate = text[start:end + 1].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

        return None

    def _find_matching_bracket(self, text: str, start: int, opener: str) -> int | None:
        """Find the matching closing bracket for a JSON array/object."""
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue
            if ch == opener:
                depth += 1
                continue
            if ch == closer:
                depth -= 1
                if depth == 0:
                    return i

        return None

    def _get_fallback_agents(self, count: int) -> list[dict[str, Any]]:
        """Return generic agents if generation fails."""
        return [
            {
                "name": f"Trader_{i}",
                "personality_traits": ["Curious", "Cautious"],
                "interests": ["General Market"],
                "beliefs": {"risk_tolerance": "Moderate", "view": "Neutral"}
            }
            for i in range(count)
        ]
