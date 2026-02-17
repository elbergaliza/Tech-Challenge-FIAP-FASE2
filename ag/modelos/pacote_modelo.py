"""
Encapsula o pacote completo do modelo (modelo treinado, metadados e scaler).
"""

from typing import Any, Optional


class PacoteModelo:
    """
    Representa o artefato modelo_completo.joblib: modelo, metadados e scaler.
    Oferece acesso tipado e métodos para uso do modelo.

    # Pacote completo para entrega
    pacote_completo = {
        # Modelo treinado
        'modelo': rf_convertido,

        # Metadados
        'metadata': {
            'data_treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sklearn_version': "1.6.1",
            'pandas_version': pd.__version__,
            'acuracia_treino': acuracia_treino,
            'acuracia_teste': acuracia_teste,
            'roc_auc': roc_auc,
            'target_name': 'HOSPITALIZ',
            'feature_names': feature_names_convert,
            'hiperparametros': {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42,
            'min_samples_leaf': 10,
            'min_samples_split': 10,
            'max_features': "sqrt",
            'n_jobs': -1
            }
        },

        # Preprocessadores (se houver)
        'scaler': scaler if 'scaler' in locals() else None,
    }

    Detalhes dos hiperparametros: veja a classe Individuo.
    """

    def __init__(
        self,
        modelo: Any,
        metadata: dict,
        scaler: Optional[Any] = None,
    ) -> None:
        self._modelo = modelo
        self._metadata = metadata
        self._scaler = scaler

    @property
    def modelo(self) -> Any:
        """Modelo treinado (ex.: RandomForestClassifier)."""
        return self._modelo

    @property
    def metadata(self) -> dict:
        """Metadados do treinamento (datas, versões, métricas, hiperparâmetros)."""
        return self._metadata

    @property
    def scaler(self) -> Optional[Any]:
        """Preprocessador scaler, se existir."""
        return self._scaler

    # Métricas
    @property
    def acuracia_treino(self) -> float:
        return self._metadata.get("acuracia_treino", 0.0)

    @property
    def acuracia_teste(self) -> float:
        return self._metadata.get("acuracia_teste", 0.0)

    @property
    def roc_auc(self) -> float:
        return self._metadata.get("roc_auc", 0.0)

    @property
    def target_name(self) -> str:
        return self._metadata.get("target_name", "")

    @property
    def feature_names(self) -> list:
        return self._metadata.get("feature_names", [])

    @property
    def data_treinamento(self) -> str:
        return self._metadata.get("data_treinamento", "")

    @property
    def hiperparametros(self) -> dict:
        return self._metadata.get("hiperparametros", {})

    def tem_scaler(self) -> bool:
        """Indica se há scaler para pré-processar entradas."""
        return self._scaler is not None

    def predict(self, X: Any) -> Any:
        """
        Predição com o modelo. Aplica scaler em X se existir.
        """
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
            f"target={self.target_name!r}, scaler={self.tem_scaler()})"
        )
