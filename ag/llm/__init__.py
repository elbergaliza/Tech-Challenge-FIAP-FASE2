import os

from .client_gemini import GeminiAdapterClient
from .generator_gemini import GeminiLaudoGenerator
from .generator_template import TemplateLaudoGenerator


def get_laudo_generator():
    provider = os.getenv("LLM_PROVIDER", "template").lower()

    if provider == "template":
        return TemplateLaudoGenerator()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment/.env")

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        max_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "20000"))
        client = GeminiAdapterClient(api_key=api_key)
        return GeminiLaudoGenerator(model=model, client=client, max_output_tokens=max_tokens)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
