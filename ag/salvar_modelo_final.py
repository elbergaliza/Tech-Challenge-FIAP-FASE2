import json
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

from ag.carga.dados_split import DadosSplit


def treinar_e_salvar_modelo_final(
    best_params: dict,
    split: DadosSplit,
    output_path: str = "data/modelo_completo.joblib",
    target_name: str = "HOSPITALIZ",
    feature_names=None,
) -> dict:
    """
    Treina um RandomForest final usando os melhores hiperparâmetros (best_params),
    avalia no teste e salva um pacote joblib com modelo + métricas + metadata.

    """

    model = RandomForestClassifier(**best_params)
    model.fit(split.X_train, split.y_train)

    proba = model.predict_proba(split.X_test)
    y_score = proba[:, 1] if proba.shape[1] == 2 else proba
    y_pred = model.predict(split.X_test)

    try:
        roc_auc = float(roc_auc_score(split.y_test, y_score, multi_class="ovr"))
    except ValueError:
        roc_auc = 0.0

    class_rep = classification_report(
        split.y_test,
        y_pred,
        labels=[0.0, 1.0],
        target_names=["Não grave", "Grave"],
        digits=4,
        zero_division="warn",
        output_dict=True,
    )

    acc_train = float(model.score(split.X_train, split.y_train))
    acc_test = float(model.score(split.X_test, split.y_test))

    if feature_names is None and hasattr(split.X_train, "columns"):
        feature_names = list(split.X_train.columns)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": "RandomForestClassifier",
        "target_name": target_name,
        "best_hyperparams": best_params,
        "metrics": {
            "roc_auc": roc_auc,
            "acuracia_treino": acc_train,
            "acuracia_teste": acc_test,
        },
        "feature_names": feature_names or [],
    }

    pacote = {
        "modelo": model,
        "aptidao": {
            "roc_auc": roc_auc,
            "acuracia_treino": acc_train,
            "acuracia_teste": acc_test,
            "classification_report": class_rep,
        },
        "metadata": metadata,
        "scaler": None, 
    }
    
    joblib.dump(pacote, output_path)

    try:
        with open("data/best_params_ag.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return metadata