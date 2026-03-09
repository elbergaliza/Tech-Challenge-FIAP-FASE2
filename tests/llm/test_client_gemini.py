from types import SimpleNamespace

import ag.llm.client_gemini as module_under_test
from ag.llm.client_gemini import GeminiAdapterClient


def test_client_gemini_sends_expected_payload(monkeypatch) -> None:
    captured: dict = {}

    class DummyModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text="  resposta teste  ")

    class DummyGenAIClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.models = DummyModels()

    monkeypatch.setattr(module_under_test.genai, "Client", DummyGenAIClient)

    client = GeminiAdapterClient(api_key="abc123")
    output = client.generate(
        "gemini-2.5-flash",
        "user prompt",
        system_prompt="sys",
        max_output_tokens=99,
        temperature=0.4,
    )

    assert output == "resposta teste"
    assert captured["api_key"] == "abc123"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["contents"] == "user prompt"
    assert captured["config"]["system_instruction"] == "sys"
    assert captured["config"]["max_output_tokens"] == 99
    assert captured["config"]["temperature"] == 0.4


def test_client_gemini_handles_none_text(monkeypatch) -> None:
    class DummyModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=None)

    class DummyGenAIClient:
        def __init__(self, api_key):
            self.models = DummyModels()

    monkeypatch.setattr(module_under_test.genai, "Client", DummyGenAIClient)

    client = GeminiAdapterClient(api_key="abc123")
    output = client.generate(
        "gemini-2.5-flash",
        "prompt",
        system_prompt="sys",
        max_output_tokens=10,
        temperature=0.1,
    )
    assert output == ""
