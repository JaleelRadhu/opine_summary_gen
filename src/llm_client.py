"""Thin wrapper around the OpenAI-compatible vLLM API."""
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text().strip()


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "default"):
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        retries: int = 3,
    ) -> Optional[str]:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [ERROR] LLM call failed after {retries} attempts: {e}")
                    return None
        return None
