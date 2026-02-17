"""
Ponto de entrada para carregar artefatos da pasta data/.
"""

from pathlib import Path
from typing import Optional, Tuple

from ag.modelos import CarregadorArtefatos, DadosSplit, DatasetProcessado, PacoteModelo

# Diretório data/ relativo à raiz do projeto (pasta que contém ag/)
_DIR_DATA = Path(__file__).resolve().parent.parent / "data"


def carregar_modelo_completo(diretorio_data: Optional[Path] = None) -> PacoteModelo:
    """Carrega o pacote completo (modelo + metadata + scaler)."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_modelo_completo()


def carregar_dataframe(diretorio_data: Optional[Path] = None) -> DatasetProcessado:
    """Carrega o DataFrame processado (CSV)."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_dataframe()


def carregar_split(diretorio_data: Optional[Path] = None) -> DadosSplit:
    """Carrega o split treino/teste."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_split()


def carregar_tudo(
    diretorio_data: Optional[Path] = None,
) -> Tuple[PacoteModelo, DatasetProcessado, DadosSplit]:
    """Carrega todos os artefatos. Retorna (pacote, dataset, split)."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_tudo()
