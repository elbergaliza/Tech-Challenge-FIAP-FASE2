"""
Encapsula o DataFrame processado (CSV carregado).
"""

from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


class DatasetProcessado:
    """
    Representa o dataset processado (ex.: DENGBR25_processado.csv).
    Expõe o DataFrame e informações derivadas (shape, colunas).
    """

    def __init__(self, df: pd.DataFrame, caminho: Optional[Path] = None) -> None:
        self._df = df
        self._caminho = caminho

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame com os dados processados."""
        return self._df

    @property
    def caminho(self) -> Optional[Path]:
        """Caminho do arquivo de origem, se conhecido."""
        return self._caminho

    @property
    def shape(self) -> tuple:
        """(n_linhas, n_colunas)."""
        return self._df.shape

    @property
    def colunas(self) -> List[str]:
        """Lista de nomes das colunas."""
        return list(self._df.columns)

    @property
    def n_linhas(self) -> int:
        return len(self._df)

    @property
    def n_colunas(self) -> int:
        return len(self._df.columns)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Primeiras n linhas."""
        return self._df.head(n)

    def coluna(self, nome: str) -> pd.Series:
        """Acesso a uma coluna pelo nome."""
        return self._df[nome]

    def __repr__(self) -> str:
        return f"DatasetProcessado(shape={self.shape}, colunas={self.n_colunas})"
