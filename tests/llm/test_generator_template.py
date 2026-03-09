from ag.llm.generator_template import TemplateLaudoGenerator
from ag.llm.laudos_type import ContextoModelo, EntradaLaudo, ResultadoModelo


def make_input(proba):
    return EntradaLaudo(
        resultado=ResultadoModelo(classe_predita=1, probabilidade_positiva=proba, limiar_decisao=0.5),
        contexto=ContextoModelo(
            nome_modelo="RandomForest",
            target_name="HOSPITALIZ",
            roc_auc_global=None,
            acuracia_teste=None,
            metadata={},
        ),
        resumo_exame={"FEBRE": "Present"},
        texto_clinico="",
    )


def test_template_has_pt_br_sections() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.8))

    assert "1) Resultado do modelo" in text
    assert "2) Interpretação" in text
    assert "3) Pontos de atenção" in text
    assert "4) Limitações" in text
    assert "Este resultado não substitui avaliação médica." in text


def test_template_risk_band_low() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.2))
    assert "nível baixo" in text


def test_template_risk_band_moderate() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.5))
    assert "nível moderado" in text


def test_template_risk_band_high() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.8))
    assert "nível alto" in text


def test_template_risk_band_undetermined() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(None))
    assert "nível indeterminado" in text
