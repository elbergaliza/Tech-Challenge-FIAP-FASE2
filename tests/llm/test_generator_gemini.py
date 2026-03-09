from ag.llm.generator_gemini import GeminiLaudoGenerator
from ag.llm.prompts import SYSTEM_PROMPT


class FakeClient:
    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(
        self,
        model: str,
        user_prompt: str,
        *,
        system_prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        return self.response


def test_build_user_prompt_contains_required_fields(entrada_laudo) -> None:
    client = FakeClient()
    generator = GeminiLaudoGenerator(model="gemini-2.5-flash", client=client, max_output_tokens=777)

    prompt = generator._build_user_prompt(entrada_laudo)

    assert "Available data:" in prompt
    assert "- Model: RandomForest" in prompt
    assert "- Target: HOSPITALIZ" in prompt
    assert "- Predicted class: 1" in prompt
    assert "- Positive class probability: 0.72" in prompt
    assert "- Decision threshold: 0.5" in prompt


def test_gerar_delegates_to_client_with_correct_params(entrada_laudo) -> None:
    client = FakeClient(response="laudo gerado")
    generator = GeminiLaudoGenerator(model="gemini-2.5-flash", client=client, max_output_tokens=1234)

    output = generator.gerar(entrada_laudo)

    assert output == "laudo gerado"
    assert len(client.calls) == 1

    call = client.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["system_prompt"] == SYSTEM_PROMPT
    assert call["max_output_tokens"] == 1234
    assert call["temperature"] == 0.2
    assert "Available data:" in call["user_prompt"]
