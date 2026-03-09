from google import genai

from .client_base import LLMClient


class GeminiAdapterClient(LLMClient):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        model: str,
        user_prompt: str,
        *,
        system_prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        resp = self._client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )
        return (resp.text or "").strip()
