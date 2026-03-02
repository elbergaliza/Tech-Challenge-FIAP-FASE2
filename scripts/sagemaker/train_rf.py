"""Treino RandomForest para SageMaker Script Mode (scikit-learn).

Espera:
- /opt/ml/input/data/train/train.csv
- /opt/ml/input/data/validation/validation.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



def _read_channel(channel: str) -> pd.DataFrame:
    path = Path(f"/opt/ml/input/data/{channel}/{channel}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Canal {channel} não encontrado: {path}")
    return pd.read_csv(path)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="HOSPITALIZ")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    train_df = _read_channel("train")
    val_df = _read_channel("validation")

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target]
    X_val = val_df.drop(columns=[args.target])
    y_val = val_df[args.target]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)
    y_score = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
    roc_auc = float(roc_auc_score(y_val, y_score, multi_class="ovr"))

    # Métrica para o HPO capturar por regex
    print(f"validation:roc_auc={roc_auc:.6f}")

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")

    metadata = {
        "target": args.target,
        "feature_names": [str(c) for c in X_train.columns],
        "roc_auc_validation": roc_auc,
        "hyperparameters": vars(args),
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
