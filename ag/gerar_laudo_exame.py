from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from ag.carregar_dados import carregar_modelo_completo, carregar_split
from ag.llm import get_laudo_generator
from ag.llm.laudos_type import EntradaLaudo, ResultadoModelo, ContextoModelo


def montar_resumo_exame(x_row, feature_names) -> Dict[str, Any]:
    resumo = {}
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(min(10, len(x_row)))]
    for i, name in enumerate(feature_names[:10]):
        resumo[name] = float(x_row[i])
    return resumo


if __name__ == "__main__":
    print("[START] generating report...")
    load_dotenv()
    import os
    print("[DEBUG] LLM_PROVIDER =", os.getenv("LLM_PROVIDER"))
    print("[DEBUG] GEMINI_API_KEY present =", bool(os.getenv("GEMINI_API_KEY")))
    print("[OK] dotenv loaded")
    ...
    print("[OK] report generated, printing...")

    pacote = carregar_modelo_completo()
    split = carregar_split()

    X = split.X_test
    x_row = X.iloc[0].values if hasattr(X, "iloc") else X[0]
    x_row = np.asarray(x_row, dtype=float)  # padroniza

    proba = None
    try:
        proba = float(pacote.predict_proba(np.array([x_row]))[0][1])
    except Exception as e:
        print(f"[WARN] predict_proba failed, falling back to predict(): {e}")

    classe = int(proba >= 0.5) if proba is not None else int(pacote.predict(np.array([x_row]))[0])

    feature_names = getattr(pacote, "feature_names", None)
    if not feature_names:
        feature_names = list(getattr(split.X_test, "columns", []))

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
        resumo_exame=montar_resumo_exame(
            x_row=x_row,
            feature_names=feature_names,
        ),
        texto_clinico=None,
    )

    gerador = get_laudo_generator()
    laudo = gerador.gerar(entrada)
    print(laudo)