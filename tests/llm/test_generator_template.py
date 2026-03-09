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


def test_template_has_english_sections() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.8))

    assert "1) Model output" in text
    assert "2) Interpretation" in text
    assert "3) Points of attention" in text
    assert "4) Limitations" in text
    assert "This text does not replace medical evaluation" in text


def test_template_risk_band_low() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.2))
    assert "probability is low" in text


def test_template_risk_band_moderate() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.5))
    assert "probability is moderate" in text


def test_template_risk_band_high() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(0.8))
    assert "probability is high" in text


def test_template_risk_band_undetermined() -> None:
    generator = TemplateLaudoGenerator()
    text = generator.gerar(make_input(None))
    assert "probability is undetermined" in text
    assert "Probability (positive class): not provided" in text
