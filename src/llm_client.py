"""
LLM client wrapper supporting OpenAI, Anthropic, and Gemini.
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    SUPPORTED_PROVIDERS = ["openai", "anthropic", "gemini", "placeholder"]

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or self._detect_provider()
        self.model = model or self._default_model()
        self.api_key = api_key or self._get_api_key()
        self._client = None

    def _detect_provider(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        return "placeholder"

    def _default_model(self) -> str:
        return {
            "openai":      "gpt-4o",
            "anthropic":   "claude-opus-4-6",
            "gemini":      "gemini-2.0-flash",
            "placeholder": "placeholder",
        }.get(self.provider, "placeholder")

    def _get_api_key(self) -> Optional[str]:
        return {
            "openai":    os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini":    os.getenv("GEMINI_API_KEY"),
        }.get(self.provider)

    def _init_client(self):
        if self._client is not None:
            return
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        else:
            self._client = "placeholder"

    def query(
        self,
        prompt: str,
        system_prompt: str = "You are an expert in causal inference and statistics. Answer precisely and show your work.",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict:
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

        elif self.provider == "gemini":
            # Gemini combines system + user prompt
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self._client.generate_content(
                full_prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            response_text = response.text

        else:
            response_text = (
                "[PLACEHOLDER] Configure an API key in .env to get real responses.\n"
                f"Prompt length: {len(prompt)} chars."
            )

        return {
            "response": response_text,
            "model": self.model,
            "provider": self.provider,
            "latency_s": round(time.time() - start, 2),
        }
