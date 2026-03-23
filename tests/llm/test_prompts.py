from ag.llm.prompts import FORMAT_INSTRUCTIONS, SYSTEM_PROMPT


def test_system_prompt_contains_safety_rules() -> None:
    assert "Do not fabricate any data" in SYSTEM_PROMPT
    assert "Do not prescribe medications" in SYSTEM_PROMPT
    assert "Do not state a definitive diagnosis" in SYSTEM_PROMPT


def test_format_instructions_mentions_pt_br_and_sections() -> None:
    assert "Brazilian Portuguese (pt-BR)" in FORMAT_INSTRUCTIONS
    assert "1) Model output" in FORMAT_INSTRUCTIONS
    assert "2) Interpretation" in FORMAT_INSTRUCTIONS
    assert "3) Points of attention" in FORMAT_INSTRUCTIONS
   