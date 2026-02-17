class Individuo:
    """
    Representa um individuo da populacao.
    Um individuo é uma possível solução para o problema.
    Um individuo é representado por um dicionário com os hiperparametros do modelo.

    O individuo é representado por um dicionário com os hiperparametros do modelo.
    O dicionário deve conter as chaves:
    - n_estimators: int
    - max_depth: int
    - random_state: int
    - min_samples_leaf: int
    - min_samples_split: int
    - max_features: int
    - n_jobs: int

    - n_estimators: int, default=100
        Numero de arvores de decisao.
        solução conhecida: 50
        range: [20, 30, 40, 50, 60, 70, 80, 90, 100]
    - max_depth: int, default=None
        Profundidade maxima da arvore de decisao.
        Se None, os nós são expandidos até que todas as folhas sejam puras ou até que todas as
            folhas contenham menos de min_samples_split amostras.
        Usar o NONE retira ele dos hiperparametros. Nao vamos utilizar esse valor.
        solução conhecida: 5
        range: [3, 4, 5, 6, 7, 8, 9, 10]
    - random_state: int, default=None
        Seed para a reproducao dos resultados.
        solução conhecida: 42
        range: [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        Considerei apenas numeros primos para evitar vieses.
    - min_samples_leaf: int or float, default=1
        Numero minimo de amostras em uma folha.
        Nao usaremos o float.
        solução conhecida: 10
        range: [6, 8, 10, 12, 14, 16, 18, 20]
        Considerei apenas numeros pares para evitar vieses.
        configure min_samples_leaf ≤ min_samples_split / 2
    - min_samples_split: int or float, default=2
        Numero minimo de amostras em um no interno.
        Nao usaremos o float.
        solução conhecida: 10
        range: [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    # - max_features: str, default="sqrt"
    #     Numero maximo de features a serem consideradas para cada split.
    #     solução conhecida: "sqrt"
    #     range: ["sqrt", "log2", "None"]
    - n_jobs: Numero de jobs a serem usados para o treinamento.
        solução conhecida: -1   (todos os cores disponiveis)
        range: [-1, 1, 2, 3, 4, 5, 6, 7]

    # TODO: acrescentar nos hiperparametros os valores STRING e FLOAT.
    # O "max_features" foi ignorado inicialmente para montarmos a solucao apenas com os hiperparametros numericos.

    """

    INDIVIDUO_DEFAULT = {
        "n_estimators": 50,
        "max_depth": 5,
        "random_state": 42,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        # "max_features": "sqrt",
        "n_jobs": -1
    }
    # INDIVIDUO_DEFAULT_STRING = {
    #     "max_features": "sqrt"
    # }
    # INDIVIDUO_DEFAULT_FLOAT = {
    #     "n_jobs": -1
    # }

    def __init__(self, hiperparametros: dict) -> None:
        """
        Inicializa o individuo.
        """
        self.hiperparametros = hiperparametros

    def __repr__(self) -> str:
        return f"Individuo(hiperparametros={self.hiperparametros})"
