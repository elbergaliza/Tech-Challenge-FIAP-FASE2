"""
Encapsula o pacote completo do modelo (modelo treinado, metadados e scaler).
"""

from typing import Any, List, Optional

from ag.classes.aptidao_modelo import AptidaoModelo


class PacoteModelo:
    """
    Representa o artefato modelo_completo.joblib.
    Oferece acesso tipado e métodos para uso do modelo e do scaler.

    Estrutura do pacote completo (modelo_completo.joblib):

    pacote_completo = {
        'modelo': modelo treinado (ex.: RandomForestClassifier),
        'aptidao': AptidaoModelo,
        'metadata': {
            'data_treinamento': str (timezone UTC-3),
            'sklearn_version': str,
            'pandas_version': str,
            'target_name': str (ex.: 'HOSPITALIZ'),
            'feature_names': list[str],
            'hiperparametros': {
                'n_estimators': int,
                'max_depth': int,
                'random_state': int,
                'min_samples_leaf': int,
                'min_samples_split': int,
                'n_jobs': int,
                'max_features': str (ex.: "sqrt"),
            },
        },
        'scaler': scaler ou None (preprocessador, se houver),
    }

    Detalhes dos hiperparâmetros: veja a classe Individuo.
    """

    def __init__(
        self,
        modelo: Any,
        aptidao: AptidaoModelo,
        metadata: dict,
        scaler: Optional[Any] = None,
    ) -> None:
        self._modelo = modelo
        self._aptidao = aptidao
        self._metadata = metadata
        self._scaler = scaler

    # -------------------------------------------------------------------------
    # Modelo e scaler
    # -------------------------------------------------------------------------

    @property
    def modelo(self) -> Any:
        """Modelo treinado (ex.: RandomForestClassifier)."""
        return self._modelo

    @property
    def scaler(self) -> Optional[Any]:
        """Preprocessador scaler, se existir."""
        return self._scaler

    # -------------------------------------------------------------------------
    # Métricas de aptidão
    # -------------------------------------------------------------------------

    @property
    def aptidao(self) -> AptidaoModelo:
        """Objeto com métricas de aptidão."""
        return self._aptidao

    @property
    def acuracia_treino(self) -> float:
        return self._aptidao.acuracia_treino

    @property
    def acuracia_teste(self) -> float:
        return self._aptidao.acuracia_teste

    @property
    def roc_auc(self) -> float:
        return self._aptidao.roc_auc

    @property
    def classification_report(self) -> dict:
        """
        Relatório de classificação por classe.

        Estrutura:
            {
                'accuracy': float,
                'Não grave': {'precision': float, 'recall': float, 'f1-score': float, 'support': int},
                'Grave': {'precision': float, 'recall': float, 'f1-score': float, 'support': int},
                'macro avg': {'precision': float, 'recall': float, 'f1-score': float, 'support': int},
                'weighted avg': {'precision': float, 'recall': float, 'f1-score': float, 'support': int},
            }
        """
        return self._aptidao.classification_report

    @property
    def accuracy(self) -> float:
        """Acurácia global do classification_report (sklearn)."""
        return self._aptidao.accuracy

    # -------------------------------------------------------------------------
    # Metadados
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """Metadados do treinamento (datas, versões, hiperparâmetros)."""
        return self._metadata

    @property
    def target_name(self) -> str:
        return self._metadata.get("target_name", "")

    @property
    def feature_names(self) -> List[str]:
        return self._metadata.get("feature_names", [])

    @property
    def data_treinamento(self) -> str:
        return self._metadata.get("data_treinamento", "")

    @property
    def sklearn_version(self) -> str:
        return self._metadata.get("sklearn_version", "")

    @property
    def pandas_version(self) -> str:
        return self._metadata.get("pandas_version", "")

    @property
    def hiperparametros(self) -> dict:
        return self._metadata.get("hiperparametros", {})

    # -------------------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------------------

    def tem_scaler(self) -> bool:
        """Indica se há scaler para pré-processar entradas."""
        return self._scaler is not None

    def predict(self, X: Any) -> Any:
        """Predição com o modelo. Aplica scaler em X se existir."""
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._modelo.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Probabilidades de classe. Aplica scaler em X se existir."""
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._modelo.predict_proba(X)

    def __repr__(self) -> str:
        return (
            f"PacoteModelo(modelo={type(self._modelo).__name__}, "
            f"target={self.target_name!r}, "
            f"roc_auc={self.roc_auc:.4f}, "
            f"scaler={self.tem_scaler()})"
        )
