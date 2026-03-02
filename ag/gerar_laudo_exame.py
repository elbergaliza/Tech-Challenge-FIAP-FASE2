from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from ag.carregar_dados import carregar_modelo_completo, carregar_split
from ag.llm import get_laudo_generator
from ag.llm.laudos_type import EntradaLaudo, ResultadoModelo, ContextoModelo


def montar_resumo_exame(x_row, feature_names) -> Dict[str, Any]:
    """Monta resumo a partir de dados normalizados (para compatibilidade)."""
    resumo = {}
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(min(10, len(x_row)))]
    for i, name in enumerate(feature_names[:10]):
        resumo[name] = float(x_row[i])
    return resumo


def montar_resumo_exame_original(dados_originais) -> Dict[str, Any]:
    """Monta resumo a partir dos dados originais do CSV (não normalizados)."""
    resumo = {}

    # Mapear valores para descrições legíveis
    for col in dados_originais.index:
        valor = dados_originais[col]

        # Interpretar colunas binárias (0/1) - sintomas e comorbidades
        if col in ['ARTRALGIA', 'ARTRITE', 'CEFALEIA', 'CONJUNTVIT', 'DOR_COSTAS',
                   'DOR_RETRO', 'EXANTEMA', 'FEBRE', 'LACO', 'LEUCOPENIA', 'MIALGIA',
                   'NAUSEA', 'PETEQUIA_N', 'VOMITO', 'ACIDO_PEPT', 'AUTO_IMUNE',
                   'DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'HIPERTENSA', 'RENAL']:
            resumo[col] = 'Present' if valor == 1.0 else 'Absent'

        # CS_SEXO: 0=Female, 1=Male (validado pelo fato de que gestante só ocorre quando sexo=0)
        elif col == 'CS_SEXO':
            resumo[col] = 'Female' if valor == 0 else 'Male'

        # CS_GESTANT: gestante (só aplicável quando sexo=0/feminino)
        elif col == 'CS_GESTANT':
            if valor == 1.0:
                resumo[col] = 'Yes'
            elif valor == 2.0:
                resumo[col] = 'No'
            else:
                resumo[col] = 'N/A or Not applicable'

        # Idade
        elif col == 'AGE_YEARS':
            resumo[col] = f"{int(valor)} years old"

        # Ignorar colunas de data e target
        elif col not in ['HOSPITALIZ', 'ANO_SIN', 'MES_SIN', 'DIA_SIN', 'DIA_SEMANA_SIN', 'NU_IDADE_N']:
            resumo[col] = valor

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

    # Carregar dados originais do CSV para enviar ao LLM
    from ag.carregar_dados import carregar_dataframe
    dataset = carregar_dataframe()

    # Pegar a primeira linha do teste (dados normalizados para predição)
    X = split.X_test
    x_row = X.iloc[0].values if hasattr(X, "iloc") else X[0]
    x_row = np.asarray(x_row, dtype=float)  # padroniza

    # Pegar também o índice original para buscar dados não normalizados
    indice_teste = X.index[0] if hasattr(X, "index") else 0
    dados_originais = dataset.df.loc[indice_teste] if hasattr(X, "index") else dataset.df.iloc[indice_teste]

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
        resumo_exame=montar_resumo_exame_original(dados_originais),
        texto_clinico=None,
    )

    gerador = get_laudo_generator()
    laudo = gerador.gerar(entrada)
    print(laudo)