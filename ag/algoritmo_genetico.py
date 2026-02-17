from ag.modelos.dataset_processado import DatasetProcessado
from ag.modelos.dados_split import DadosSplit
from ag.modelos.individuo import Individuo


def gerar_populacao_inicial(dataset: DatasetProcessado, split: DadosSplit) -> list[Individuo]:
    """
    Gera a população inicial.
    """
    return []  # TODO: implementar a geracao da populacao inicial
