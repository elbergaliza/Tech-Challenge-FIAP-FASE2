from ag.avaliar_laudos import DISCLAIMER, auto_checks


def test_auto_checks_valid_pt_br_report() -> None:
    laudo = f"""\
1) Resultado do modelo
texto

2) Interpretação
texto

3) Pontos de atenção
texto

4) Limitações
texto
{DISCLAIMER}
"""
    checks = auto_checks(laudo)
    assert checks["has_section_1"] is True
    assert checks["has_section_2"] is True
    assert checks["has_section_3"] is True
    assert checks["mentions_limitations"] is True
    assert checks["has_disclaimer"] is True


def test_auto_checks_missing_section_2() -> None:
    laudo = f"""\
1) Resultado do modelo
texto

3) Pontos de atenção
texto

4) Limitações
texto
{DISCLAIMER}
"""
    checks = auto_checks(laudo)
    assert checks["has_section_2"] is False
