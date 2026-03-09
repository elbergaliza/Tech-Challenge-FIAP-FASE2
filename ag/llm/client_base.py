from typing import Protocol


class LLMClient(Protocol):
    def generate(
        self,
        model: str,
        user_prompt: str,
        *,
        system_prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        ...
