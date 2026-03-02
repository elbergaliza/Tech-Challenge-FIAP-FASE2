import os

from .generator_base import LaudoGenerator
from .generator_template import TemplateLaudoGenerator

def get_laudo_generator():
    provider = os.getenv("LLM_PROVIDER", "template").lower()

    if provider == "gemini":
        from .generator_gemini import GeminiLaudoGenerator
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        max_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "20000"))
        return GeminiLaudoGenerator(model=model, max_output_tokens=max_tokens)

    from .generator_template import TemplateLaudoGenerator
    return TemplateLaudoGenerator()