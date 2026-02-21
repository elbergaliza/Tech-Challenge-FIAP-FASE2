"""
Classes para manipulação dos artefatos carregados (modelo, dataset, split).
"""

from ag.modelos.aptidao_modelo import AptidaoModelo
from ag.modelos.carregador_artefatos import CarregadorArtefatos
from ag.modelos.dados_split import DadosSplit
from ag.modelos.dataset_processado import DatasetProcessado
from ag.modelos.individuo import Individuo
from ag.modelos.modelo import PacoteModelo
from ag.modelos.populacao import Populacao

__all__ = [
    "AptidaoModelo",
    "CarregadorArtefatos",
    "DadosSplit",
    "DatasetProcessado",
    "Individuo",
    "PacoteModelo",
    "Populacao",
]
