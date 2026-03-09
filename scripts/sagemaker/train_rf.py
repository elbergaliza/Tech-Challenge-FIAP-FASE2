"""Treino RandomForest para SageMaker Script Mode (scikit-learn).

Espera:
- /opt/ml/input/data/train/train.csv
- /opt/ml/input/data/validation/validation.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)



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
    t0 = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - t0

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    y_score = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba

    try:
        roc_auc = float(roc_auc_score(y_val, y_score, multi_class="ovr"))
    except ValueError:
        roc_auc = 0.0
    accuracy = float(accuracy_score(y_val, y_pred))
    # average="binary" assume classe positiva = 1 (hospitalização grave)
    precision = float(precision_score(y_val, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_val, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average="binary", zero_division=0))
    conf_matrix = confusion_matrix(y_val, y_pred).tolist()

    clf_report = classification_report(
        y_val,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    # Métricas para o HPO capturar por regex (uma por linha)
    print(f"validation:roc_auc={roc_auc:.6f}")
    print(f"validation:accuracy={accuracy:.6f}")
    print(f"validation:precision={precision:.6f}")
    print(f"validation:recall={recall:.6f}")
    print(f"validation:f1={f1:.6f}")

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")

    metadata = {
        "target": args.target,
        "feature_names": [str(c) for c in X_train.columns],
        "hyperparameters": vars(args),
        "train_duration_seconds": round(train_duration, 3),
        "aptidao": {
            "roc_auc_validation": roc_auc,
            "accuracy_validation": accuracy,
            "precision_validation": precision,
            "recall_validation": recall,
            "f1_validation": f1,
        },
        "confusion_matrix": conf_matrix,
        "classification_report": clf_report,
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
