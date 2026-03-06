"""
LLM client wrapper.

Placeholder for LLM API calls. Supports OpenAI and Anthropic APIs.
Fill in your API key in .env to activate.
"""

import os
import json
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified LLM client. Configure via .env file or constructor args.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or self._detect_provider()
        self.model = model or os.getenv("LLM_MODEL", self._default_model())
        self.api_key = api_key or self._get_api_key()
        self._client = None

    def _detect_provider(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "placeholder"

    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "placeholder": "placeholder",
        }
        return defaults.get(self.provider, "placeholder")

    def _get_api_key(self) -> Optional[str]:
        keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = keys.get(self.provider)
        return os.getenv(env_var) if env_var else None

    def _init_client(self):
        if self._client is not None:
            return

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self._client = "placeholder"

    def query(
        self,
        prompt: str,
        system_prompt: str = "You are an expert in causal inference and statistics. Answer precisely and show your work.",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Send a query to the LLM and return the response.

        Returns dict with:
          - response: the LLM's text response
          - model: model used
          - provider: provider used
          - latency_s: response time in seconds
        """
        self._init_client()
        start = time.time()

        if self.provider == "openai":
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_text = completion.choices[0].message.content

        elif self.provider == "anthropic":
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            response_text = message.content[0].text

        else:
            # Placeholder: return a mock response for testing the pipeline
            response_text = (
                "[PLACEHOLDER RESPONSE]\n"
                "This is a mock response. Configure your LLM API key in .env "
                "to get real responses.\n\n"
                f"Received prompt of length {len(prompt)} characters."
            )

        latency = time.time() - start

        return {
            "response": response_text,
            "model": self.model,
            "provider": self.provider,
            "latency_s": round(latency, 2),
        }
