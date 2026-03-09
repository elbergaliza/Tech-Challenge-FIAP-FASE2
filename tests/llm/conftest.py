import pytest

from ag.llm.laudos_type import ContextoModelo, EntradaLaudo, ResultadoModelo


@pytest.fixture
def entrada_laudo() -> EntradaLaudo:
    return EntradaLaudo(
        resultado=ResultadoModelo(
            classe_predita=1,
            probabilidade_positiva=0.72,
            limiar_decisao=0.5,
        ),
        contexto=ContextoModelo(
            nome_modelo="RandomForest",
            target_name="HOSPITALIZ",
            roc_auc_global=0.88,
            acuracia_teste=0.81,
            metadata={"origem": "teste"},
        ),
        resumo_exame={
            "FEBRE": "Present",
            "AGE_YEARS": "34 years old",
        },
        texto_clinico="Paciente com mialgia e cefaleia.",
    )
