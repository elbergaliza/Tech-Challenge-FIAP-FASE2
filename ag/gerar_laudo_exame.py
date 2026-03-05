from typing import Dict, Any
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def load_dotenv_upwards(filename: str = ".env", max_depth: int = 6) -> Path | None:
    start = Path(__file__).resolve().parent
    cur = start
    for _ in range(max_depth):
        candidate = cur / filename
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=True)
            return candidate
        cur = cur.parent
    return None

ENV_PATH = load_dotenv_upwards()

from ag.carga import carregar_modelo_completo, carregar_split, carregar_dataframe, carregar_dados
from ag.llm import get_laudo_generator
from ag.llm.laudos_type import EntradaLaudo, ResultadoModelo, ContextoModelo


def montar_resumo_exame_original(dados_originais) -> Dict[str, Any]:
    resumo = {}
    for col in dados_originais.index:
        valor = dados_originais[col]

        if col in [
            'ARTRALGIA', 'ARTRITE', 'CEFALEIA', 'CONJUNTVIT', 'DOR_COSTAS',
            'DOR_RETRO', 'EXANTEMA', 'FEBRE', 'LACO', 'LEUCOPENIA', 'MIALGIA',
            'NAUSEA', 'PETEQUIA_N', 'VOMITO', 'ACIDO_PEPT', 'AUTO_IMUNE',
            'DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'HIPERTENSA', 'RENAL'
        ]:
            resumo[col] = "Present" if float(valor) == 1.0 else "Absent"

        elif col == 'CS_SEXO':
            resumo[col] = "Female" if float(valor) == 0.0 else "Male"

        elif col == 'CS_GESTANT':
            if float(valor) == 1.0:
                resumo[col] = "Yes"
            elif float(valor) == 2.0:
                resumo[col] = "No"
            else:
                resumo[col] = "N/A"

        elif col == 'AGE_YEARS':
            resumo[col] = f"{int(valor)} years old"

        elif col not in ['HOSPITALIZ', 'ANO_SIN', 'MES_SIN', 'DIA_SIN', 'DIA_SEMANA_SIN', 'NU_IDADE_N']:
            resumo[col] = valor

    return resumo


if __name__ == "__main__":
    print("[DEBUG] .env found at:", ENV_PATH)
    print("[DEBUG] LLM_PROVIDER =", os.getenv("LLM_PROVIDER"))
    print("[DEBUG] GEMINI_API_KEY present =", bool(os.getenv("GEMINI_API_KEY")))

    pacote = carregar_modelo_completo()
    split = carregar_split()
    dataset = carregar_dataframe()

    X = split.X_test
    x_row = X.iloc[0].values if hasattr(X, "iloc") else X[0]
    x_row = np.asarray(x_row, dtype=float)

    indice_teste = X.index[0] if hasattr(X, "index") else 0

    df = getattr(dataset, "df", None)
    if df is None and isinstance(dataset, dict):
        df = dataset.get("df") or dataset.get("dataframe")
    if df is None:
        raise RuntimeError("carregar_dataframe() did not return an object with .df (or dict with 'df').")

    dados_originais = df.loc[indice_teste] if hasattr(X, "index") else df.iloc[indice_teste]

    proba = None
    try:
        proba = float(pacote.predict_proba(np.array([x_row]))[0][1])
    except Exception as e:
        print(f"[WARN] predict_proba failed, falling back to predict(): {e}")

    classe = int(proba >= 0.5) if proba is not None else int(pacote.predict(np.array([x_row]))[0])

    entrada = EntradaLaudo(
        resultado=ResultadoModelo(
            classe_predita=classe,
            probabilidade_positiva=proba,
            limiar_decisao=0.5,
        ),
        contexto=ContextoModelo(
            nome_modelo="RandomForest",
            target_name=getattr(pacote, "target_name", "unknown_target"),
            roc_auc_global=getattr(pacote, "roc_auc", None),
            acuracia_teste=getattr(pacote, "acuracia_teste", None),
            metadata=getattr(pacote, "metadata", {}) or {},
        ),
        resumo_exame=montar_resumo_exame_original(dados_originais),
        texto_clinico="",
    )

    gerador = get_laudo_generator()
    laudo = gerador.gerar(entrada)
    print(laudo)