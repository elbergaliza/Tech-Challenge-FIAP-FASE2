"""
Encapsula o split treino/teste (X_train, X_test, y_train, y_test).
"""

from typing import Any, Tuple


class DadosSplit:
    """
    Representa o artefato dados_split.joblib.
    Expõe conjuntos de treino e teste com propriedades e informações de tamanho.
    """

    def __init__(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
    ) -> None:
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

    @property
    def X_train(self) -> Any:
        return self._X_train

    @property
    def X_test(self) -> Any:
        return self._X_test

    @property
    def y_train(self) -> Any:
        return self._y_train

    @property
    def y_test(self) -> Any:
        return self._y_test

    @property
    def n_amostras_treino(self) -> int:
        if hasattr(self._X_train, "shape"):
            return self._X_train.shape[0]
        return len(self._X_train)

    @property
    def n_amostras_teste(self) -> int:
        if hasattr(self._X_test, "shape"):
            return self._X_test.shape[0]
        return len(self._X_test)

    def shape_treino(self) -> Tuple[int, ...]:
        """Retorna shape de X_train (n_amostras, n_features)."""
        if hasattr(self._X_train, "shape"):
            return tuple(self._X_train.shape)
        return (self.n_amostras_treino,)

    def shape_teste(self) -> Tuple[int, ...]:
        """Retorna shape de X_test."""
        if hasattr(self._X_test, "shape"):
            return tuple(self._X_test.shape)
        return (self.n_amostras_teste,)

    def __repr__(self) -> str:
        return (
            f"DadosSplit(treino={self.n_amostras_treino} amostras, "
            f"teste={self.n_amostras_teste} amostras)"
        )
