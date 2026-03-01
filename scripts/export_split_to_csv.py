#!/usr/bin/env python3
"""Exporta data/dados_split.joblib para CSVs usados no SageMaker HPO.

Saídas:
- train.csv (train interno para HPO)
- validation.csv (validação para objetivo do HPO)
- test_features.csv (features para batch/final)
- test_labels.csv (rótulos do teste final)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


def _to_dataframe(x, feature_names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if feature_names and len(feature_names) == x.shape[1]:
        return pd.DataFrame(x, columns=feature_names)
    cols = [f"f{i}" for i in range(x.shape[1])]
    return pd.DataFrame(x, columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta split joblib para CSVs do SageMaker")
    parser.add_argument("--split", default="data/dados_split.joblib", help="Arquivo de split joblib")
    parser.add_argument("--model", default="data/modelo_completo.joblib", help="Modelo para descobrir nomes de colunas")
    parser.add_argument("--target", default="HOSPITALIZ", help="Nome da coluna target")
    parser.add_argument("--val-size", type=float, default=0.2, help="Percentual de validação do X_train")
    parser.add_argument("--seed", type=int, default=42, help="Seed para split interno")
    parser.add_argument("--out-dir", default="data/sagemaker", help="Pasta de saída")
    args = parser.parse_args()

    split = joblib.load(args.split)
    model_pkg = joblib.load(args.model)

    feature_names = model_pkg.get("metadata", {}).get("feature_names")
    X_train = _to_dataframe(split["X_train"], feature_names)
    X_test = _to_dataframe(split["X_test"], feature_names)
    y_train = pd.Series(split["y_train"]).reset_index(drop=True)
    y_test = pd.Series(split["y_test"]).reset_index(drop=True)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_tr.copy()
    train_df[args.target] = y_tr.values
    val_df = X_val.copy()
    val_df[args.target] = y_val.values

    train_path = out_dir / "train.csv"
    val_path = out_dir / "validation.csv"
    test_features_path = out_dir / "test_features.csv"
    test_labels_path = out_dir / "test_labels.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    X_test.to_csv(test_features_path, index=False)
    pd.DataFrame({args.target: y_test.values}).to_csv(test_labels_path, index=False)

    print("Arquivos gerados:")
    print(f"- {train_path} ({train_df.shape[0]} linhas)")
    print(f"- {val_path} ({val_df.shape[0]} linhas)")
    print(f"- {test_features_path} ({X_test.shape[0]} linhas)")
    print(f"- {test_labels_path} ({y_test.shape[0]} linhas)")


if __name__ == "__main__":
    main()
