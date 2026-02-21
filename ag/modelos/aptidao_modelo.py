"""
Estrutura de aptidao do modelo treinado.
"""

from __future__ import annotations

from typing import Any, Dict


_DEFAULT_REPORT: Dict[str, Any] = {}


class AptidaoModelo:
    """
    Representa as metricas de aptidao de um modelo treinado.

    Estrutura esperada:
        {
            "acuracia_treino": float,
            "acuracia_teste": float,
            "roc_auc": float,
            "classification_report": {
                "accuracy": float,
                "Não grave": {"precision": float, "recall": float, "f1-score": float, "support": int},
                "Grave": {"precision": float, "recall": float, "f1-score": float, "support": int},
                "macro avg": {"precision": float, "recall": float, "f1-score": float, "support": int},
                "weighted avg": {"precision": float, "recall": float, "f1-score": float, "support": int},
            },       
        }
    """

    def __init__(
        self,
        acuracia_treino: float = 0.0,
        acuracia_teste: float = 0.0,
        roc_auc: float = 0.0,
        classification_report: Dict[str, Any] = _DEFAULT_REPORT,
    ) -> None:
        self._acuracia_treino = acuracia_treino
        self._acuracia_teste = acuracia_teste
        self._roc_auc = roc_auc
        self._classification_report = dict(classification_report)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AptidaoModelo":
        """Cria AptidaoModelo a partir de um dicionario."""
        return cls(
            acuracia_treino=float(data.get("acuracia_treino", 0.0)),
            acuracia_teste=float(data.get("acuracia_teste", 0.0)),
            roc_auc=float(data.get("roc_auc", 0.0)),
            classification_report=data.get("classification_report", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Retorna a estrutura de aptidao como dicionario."""
        return {
            "acuracia_treino": self._acuracia_treino,
            "acuracia_teste": self._acuracia_teste,
            "roc_auc": self._roc_auc,
            "classification_report": self._classification_report,
        }

    @property
    def acuracia_treino(self) -> float:
        return self._acuracia_treino

    @property
    def acuracia_teste(self) -> float:
        return self._acuracia_teste

    @property
    def roc_auc(self) -> float:
        return self._roc_auc

    @property
    def classification_report(self) -> Dict[str, Any]:
        return self._classification_report

    @property
    def accuracy(self) -> float:
        """Acuracia global do classification_report (sklearn)."""
        valor = self._classification_report.get("accuracy", 0.0)
        try:
            return float(valor)
        except (TypeError, ValueError):
            return 0.0

    def __repr__(self) -> str:
        return (
            "AptidaoModelo("
            f"acuracia_treino={self._acuracia_treino:.4f}, "
            f"acuracia_teste={self._acuracia_teste:.4f}, "
            f"roc_auc={self._roc_auc:.4f})"
        )
