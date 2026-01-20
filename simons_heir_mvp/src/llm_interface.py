"""LLM Interface module for Ollama API communication.

Provides abstraction layer for interacting with local Llama model via Ollama.
"""

import json
import logging
import time
from typing import Any

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

from .config import (
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM_DEFAULT_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LlamaInterfaceError(Exception):
    """Base exception for LlamaInterface errors."""
    pass


class LlamaConnectionError(LlamaInterfaceError):
    """Raised when connection to Ollama fails."""
    pass


class LlamaResponseError(LlamaInterfaceError):
    """Raised when Ollama returns an invalid response."""
    pass


class LlamaInterface:
    """Interface for communicating with Llama model via Ollama API.
    
    Handles request/response cycle with retry logic and error handling.
    
    Attributes:
        model_name: Name of the Ollama model to use.
        endpoint: URL of the Ollama generate endpoint.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
    """
    
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        endpoint: str = OLLAMA_GENERATE_ENDPOINT,
        timeout: int = LLM_TIMEOUT_SECONDS,
        max_retries: int = LLM_MAX_RETRIES,
        mock_mode: bool = False,
    ) -> None:
        """Initialize LlamaInterface.
        
        Args:
            model_name: Name of the Ollama model (default: llama3.1:8b).
            endpoint: Ollama API endpoint URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts on failure.
            mock_mode: Whether to run in mock mode (no API calls).
        """
        self.model_name = model_name
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self._session = requests.Session()
    
    def generate(
        self,
        prompt: str,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> str:
        """Generate text completion from the LLM.
        
        Args:
            prompt: The input prompt to send to the model.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate (None for model default).
            stream: Whether to use streaming response.
            
        Returns:
            Generated text response from the model.
            
        Raises:
            LlamaConnectionError: If connection to Ollama fails after retries.
            LlamaResponseError: If response parsing fails.
        """
        if self.mock_mode:
            logger.info("Mock mode: Returning dummy response")
            return "ACTION: HOLD\nCONTENT: Monitoring the situation (Mock Decision)."

        payload = self._build_payload(prompt, temperature, max_tokens, stream)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._make_request(payload)
                return self._handle_response(response)
            except (ConnectionError, Timeout) as e:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise LlamaConnectionError(
                        f"Failed to connect to Ollama after {self.max_retries} attempts: {e}"
                    ) from e
            except RequestException as e:
                raise LlamaConnectionError(f"Request failed: {e}") from e
        
        raise LlamaConnectionError("Unexpected error in retry loop")
    
    def _build_payload(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the request payload for Ollama API.
        
        Args:
            prompt: Input prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: Streaming mode flag.
            
        Returns:
            Dictionary payload for the API request.
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        return payload
    
    def _make_request(self, payload: dict[str, Any]) -> requests.Response:
        """Send POST request to Ollama API.
        
        Args:
            payload: Request payload dictionary.
            
        Returns:
            Response object from the API.
            
        Raises:
            RequestException: If the request fails.
        """
        response = self._session.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response
    
    def _handle_response(self, response: requests.Response) -> str:
        """Parse and extract text from Ollama API response.
        
        Args:
            response: Response object from Ollama API.
            
        Returns:
            Extracted generated text.
            
        Raises:
            LlamaResponseError: If response parsing fails.
        """
        try:
            data = response.json()
            
            if "response" in data:
                return data["response"].strip()
            elif "error" in data:
                raise LlamaResponseError(f"Ollama error: {data['error']}")
            else:
                raise LlamaResponseError(
                    f"Unexpected response format: {json.dumps(data)[:200]}"
                )
        except json.JSONDecodeError as e:
            raise LlamaResponseError(f"Failed to parse JSON response: {e}") from e
    
    def health_check(self) -> bool:
        """Check if Ollama service is available.
        
        Returns:
            True if Ollama is responsive, False otherwise.
        """
        if self.mock_mode:
            return True

        try:
            response = self._session.get(
                self.endpoint.replace("/api/generate", "/api/tags"),
                timeout=5,
            )
            return response.status_code == 200
        except RequestException:
            return False
    
    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()
    
    def __enter__(self) -> "LlamaInterface":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
