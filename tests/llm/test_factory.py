import ag.llm as llm_module
from ag.llm import get_laudo_generator
from ag.llm.generator_gemini import GeminiLaudoGenerator
from ag.llm.generator_template import TemplateLaudoGenerator


def test_factory_defaults_to_template(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    generator = get_laudo_generator()
    assert isinstance(generator, TemplateLaudoGenerator)


def test_factory_template_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "template")
    generator = get_laudo_generator()
    assert isinstance(generator, TemplateLaudoGenerator)


def test_factory_gemini_without_key_raises(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    try:
        get_laudo_generator()
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Missing GEMINI_API_KEY" in str(exc)


def test_factory_gemini_with_key_returns_generator(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "111")

    class DummyAdapter:
        def __init__(self, api_key):
            self.api_key = api_key

        def generate(self, *args, **kwargs):
            return "ok"

    monkeypatch.setattr(llm_module, "GeminiAdapterClient", DummyAdapter)

    generator = get_laudo_generator()
    assert isinstance(generator, GeminiLaudoGenerator)
    assert generator.max_output_tokens == 111


def test_factory_invalid_provider_raises(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "invalid-provider")
    try:
        get_laudo_generator()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Unsupported LLM_PROVIDER" in str(exc)
