import re
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from dotenv import load_dotenv

from ag.carga import carregar_modelo_completo, carregar_split, carregar_dataframe
from ag.llm import get_laudo_generator
from ag.llm.laudos_type import EntradaLaudo, ResultadoModelo, ContextoModelo

load_dotenv()

DISCLAIMER = "Este resultado não substitui avaliação médica."


def montar_resumo_exame_original(dados_originais) -> Dict[str, Any]:
    resumo = {}
    for col in dados_originais.index:
        valor = dados_originais[col]
        if col in [
            'ARTRALGIA','ARTRITE','CEFALEIA','CONJUNTVIT','DOR_COSTAS',
            'DOR_RETRO','EXANTEMA','FEBRE','LACO','LEUCOPENIA','MIALGIA',
            'NAUSEA','PETEQUIA_N','VOMITO','ACIDO_PEPT','AUTO_IMUNE',
            'DIABETES','HEMATOLOG','HEPATOPAT','HIPERTENSA','RENAL'
        ]:
            resumo[col] = "Present" if float(valor) == 1.0 else "Absent"
        elif col == 'CS_SEXO':
            resumo[col] = "Female" if float(valor) == 0.0 else "Male"
        elif col == 'CS_GESTANT':
            resumo[col] = "Yes" if float(valor) == 1.0 else ("No" if float(valor) == 2.0 else "N/A")
        elif col == 'AGE_YEARS':
            resumo[col] = f"{int(valor)} years old"
        elif col not in ['HOSPITALIZ','ANO_SIN','MES_SIN','DIA_SIN','DIA_SEMANA_SIN','NU_IDADE_N']:
            resumo[col] = valor
    return resumo


def auto_checks(laudo: str) -> Dict[str, Any]:
    has_1 = bool(re.search(r"(?im)^1\)\s+Resultado do modelo", laudo))
    has_2 = bool(re.search(r"(?im)^2\)\s+Interpretação", laudo))
    has_3 = bool(re.search(r"(?im)^3\)\s+Pontos de atenção", laudo))
    has_limitations = bool(re.search(r"(?im)limita", laudo))  
    has_disclaimer = DISCLAIMER in laudo

    return {
        "has_section_1": has_1,
        "has_section_2": has_2,
        "has_section_3": has_3,
        "has_disclaimer": has_disclaimer,
        "mentions_limitations": has_limitations,
        "len_chars": len(laudo),
    }


if __name__ == "__main__":
    pacote = carregar_modelo_completo()
    split = carregar_split()
    dataset = carregar_dataframe()
    gerador = get_laudo_generator()

    X = split.X_test
    df = getattr(dataset, "df", None)
    if df is None and isinstance(dataset, dict):
        df = dataset.get("df") or dataset.get("dataframe")
    if df is None:
        raise RuntimeError("Dataset does not provide df.")

    out_dir = Path("outputs/laudos")
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of test cases to evaluate")
    args = parser.parse_args()

    N = min(args.n, len(X))
    rows: List[Dict[str, Any]] = []

    for i in range(N):
        x_row = X.iloc[i].values if hasattr(X, "iloc") else X[i]
        x_row = np.asarray(x_row, dtype=float)
        idx = X.index[i] if hasattr(X, "index") else i
        dados_originais = df.loc[idx] if hasattr(X, "index") else df.iloc[idx]

        proba = float(pacote.predict_proba(np.array([x_row]))[0][1])
        classe = int(proba >= 0.5)

        entrada = EntradaLaudo(
            resultado=ResultadoModelo(classe_predita=classe, probabilidade_positiva=proba, limiar_decisao=0.5),
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

        laudo = gerador.gerar(entrada)
        checks = auto_checks(laudo)

        (out_dir / f"laudo_{i:03d}.md").write_text(laudo, encoding="utf-8")

        rows.append({
            "case": i,
            "index": str(idx),
            "proba": proba,
            "classe": classe,
            **checks
        })

    csv_path = Path("outputs/laudos_summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Saved {N} reports to {out_dir}")
    print(f"[OK] Saved summary to {csv_path}")