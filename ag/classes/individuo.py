import copy
import random
from typing import Optional

from ag.classes.aptidao_modelo import AptidaoModelo


class Individuo:
    """
    Representa um individuo da populacao.
    Um individuo é uma possível solução para o problema.
    Um individuo é representado por um dicionário com os hiperparametros do modelo.

    O individuo é representado por um dicionário com os hiperparametros do modelo.
    O dicionário apresenta as chaves:

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
    - n_jobs: Numero de jobs a serem usados para o treinamento.
        solução conhecida: -1   (todos os cores disponiveis)
        range: [-1, 1, 2, 3, 4, 5, 6, 7]

    # O "max_features" foi ignorado inicialmente para montarmos a solucao apenas com os hiperparametros numericos.
    # - max_features: str, default="sqrt"
    #     Numero maximo de features a serem consideradas para cada split.
    #     solução conhecida: "sqrt"
    #     range: ["sqrt", "log2", "None"]

    Attributes:
        hiperparametros: Dicionário com os valores dos hiperparâmetros.
        aptidao: AptidaoModelo do individuo.
        ESPACOS_BUSCA: Dicionário com os espaços de busca para os hiperparâmetros.
        Os valores de cada hiperparametro foram escolhidos analisando a documentação da 
            classe RandomForestClassifier.

        INDIVIDUO_DEFAULT: Dicionário com os valores padrão para os hiperparâmetros. 
            Esses valores foram utilizados no treinamento do modelo. Portanto, são os que serão usados 
            para criar o individuo default.
        #INDIVIDUO_DEFAULT_STRING: Dicionário com os valores padrão para os hiperparâmetros STRING.
        #INDIVIDUO_DEFAULT_FLOAT: Dicionário com os valores padrão para os hiperparâmetros FLOAT.
    """

    INDIVIDUO_DEFAULT = {
        "n_estimators": 50,
        "max_depth": 5,
        "random_state": 42,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        "n_jobs": -1
    }

    ESPACOS_BUSCA: dict[str, list] = {
        "n_estimators":      list(range(20, 100)),
        "max_depth":         list(range(3, 50)),
        "random_state":      [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
        "min_samples_leaf":  list(range(6, 50)),
        "min_samples_split": list(range(12, 100)),
        "n_jobs":            [-1, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    def __init__(self, hiperparametros: dict) -> None:
        """
        Inicializa o individuo.

        Args:
            hiperparametros: Dicionário com os valores dos hiperparâmetros.
        """
        self.hiperparametros = hiperparametros
        self.aptidao: Optional[AptidaoModelo] = None

    # -------------------------------------------------------------------------
    # Métodos de fábrica (class methods)
    # -------------------------------------------------------------------------

    @classmethod
    def gerar_aleatorio(cls) -> "Individuo":
        """
        Cria um indivíduo com hiperparâmetros sorteados aleatoriamente dos
        espaços de busca definidos em ESPACOS_BUSCA.

        Garante a restrição: min_samples_leaf ≤ min_samples_split / 2.

        Returns:
            Novo Individuo com genes aleatórios e válidos.
        """
        hiperparametros: dict = {}
        for gene, valores in cls.ESPACOS_BUSCA.items():
            hiperparametros[gene] = random.choice(valores)

        individuo = cls(hiperparametros)
        individuo._corrigir_constraints()
        return individuo

    @classmethod
    def default(cls) -> "Individuo":
        """
        Cria um indivíduo com os hiperparâmetros padrão (solução de referência
        definida em INDIVIDUO_DEFAULT).

        Returns:
            Novo Individuo com os hiperparâmetros padrão.
        """
        return cls(copy.deepcopy(cls.INDIVIDUO_DEFAULT))

    # -------------------------------------------------------------------------
    # Operadores genéticos
    # -------------------------------------------------------------------------

    def cruzar(self, outro: "Individuo") -> tuple["Individuo", "Individuo"]:
        """
        Operador de cruzamento uniforme (crossover) entre dois indivíduos.

        Para cada gene, sorteia com igual probabilidade se o filho herda o
        gene deste indivíduo ou do parceiro. Dois filhos complementares são
        gerados garantindo diversidade.

        Garante a restrição: min_samples_leaf ≤ min_samples_split / 2.

        Args:
            outro: Indivíduo parceiro para o cruzamento.

        Returns:
            Tupla (filho1, filho2) com os dois descendentes gerados.
        """
        genes = list(self.ESPACOS_BUSCA.keys())
        filho1_hp: dict = {}
        filho2_hp: dict = {}

        for gene in genes:
            if random.random() < 0.5:
                filho1_hp[gene] = self.hiperparametros[gene]
                filho2_hp[gene] = outro.hiperparametros[gene]
            else:
                filho1_hp[gene] = outro.hiperparametros[gene]
                filho2_hp[gene] = self.hiperparametros[gene]

        filho1 = Individuo(filho1_hp)
        filho2 = Individuo(filho2_hp)

        filho1._corrigir_constraints()
        filho2._corrigir_constraints()

        return filho1, filho2

    def mutar(self, taxa_mutacao: float = 0.1) -> "Individuo":
        """
        Operador de mutação por substituição aleatória de genes.

        Cada gene é substituído, com probabilidade ``taxa_mutacao``, por um
        valor sorteado aleatoriamente do respectivo espaço de busca.

        Garante a restrição: min_samples_leaf ≤ min_samples_split / 2.

        Args:
            taxa_mutacao: Probabilidade de mutação por gene. Deve estar em
                          [0.0, 1.0]. Padrão: 0.1 (10 %).

        Returns:
            Novo Individuo mutado (o original não é modificado).
        """
        novo_hp = copy.deepcopy(self.hiperparametros)

        for gene, valores in self.ESPACOS_BUSCA.items():
            if random.random() < taxa_mutacao:
                novo_hp[gene] = random.choice(valores)

        mutante = Individuo(novo_hp)
        mutante._corrigir_constraints()
        return mutante

    # -------------------------------------------------------------------------
    # Validação e restrições
    # -------------------------------------------------------------------------

    def _corrigir_constraints(self) -> None:
        """
        Corrige a restrição: min_samples_leaf ≤ min_samples_split / 2.

        Se violada, reduz ``min_samples_leaf`` ao maior valor válido presente
        no espaço de busca para o valor atual de ``min_samples_split``.
        Como último recurso, eleva ``min_samples_split`` ao máximo disponível.
        """
        limite = self.hiperparametros["min_samples_split"] / 2
        valores_validos = [
            v for v in self.ESPACOS_BUSCA["min_samples_leaf"] if v <= limite
        ]

        if not valores_validos:
            # Eleva min_samples_split ao máximo para abrir espaço
            self.hiperparametros["min_samples_split"] = max(
                self.ESPACOS_BUSCA["min_samples_split"]
            )
            limite = self.hiperparametros["min_samples_split"] / 2
            valores_validos = [
                v for v in self.ESPACOS_BUSCA["min_samples_leaf"] if v <= limite
            ]

        if self.hiperparametros["min_samples_leaf"] > limite:
            self.hiperparametros["min_samples_leaf"] = max(valores_validos)

    def eh_valido(self) -> bool:
        """
        Verifica se o indivíduo é completamente válido:
        - Todos os genes estão presentes.
        - Os valores pertencem aos respectivos espaços de busca.
        - A restrição min_samples_leaf ≤ min_samples_split / 2 é satisfeita.

        Returns:
            True se o indivíduo for válido, False caso contrário.
        """
        for gene, valores in self.ESPACOS_BUSCA.items():
            if self.hiperparametros.get(gene) not in valores:
                return False

        if (
            self.hiperparametros["min_samples_leaf"]
            > self.hiperparametros["min_samples_split"] / 2
        ):
            return False

        return True

    # -------------------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------------------

    def copiar(self) -> "Individuo":
        """
        Cria uma cópia profunda do indivíduo, preservando o valor de aptidão.

        Returns:
            Novo Individuo idêntico ao original.
        """
        clone = Individuo(copy.deepcopy(self.hiperparametros))
        clone.aptidao = self.aptidao
        return clone

    def para_dict(self) -> dict:
        """
        Retorna os hiperparâmetros como dicionário simples (cópia segura).

        Returns:
            Dicionário com os hiperparâmetros do indivíduo.
        """
        return copy.deepcopy(self.hiperparametros)

    # -------------------------------------------------------------------------
    # Comparação e representação
    # -------------------------------------------------------------------------

    def __eq__(self, outro: object) -> bool:
        if not isinstance(outro, Individuo):
            return NotImplemented
        return self.hiperparametros == outro.hiperparametros

    def __lt__(self, outro: "Individuo") -> bool:
        """Permite ordenar indivíduos pela aptidão (maior aptidão = melhor)."""
        if self.aptidao is None or outro.aptidao is None:
            raise ValueError(
                "Não é possível comparar indivíduos sem aptidão calculada."
            )
        return self.aptidao.roc_auc < outro.aptidao.roc_auc

    def __repr__(self) -> str:
        aptidao_str = (
            f", aptidao={self.aptidao.roc_auc:.4f}"
            if self.aptidao is not None
            else ""
        )
        return f"Individuo(hiperparametros={self.hiperparametros}{aptidao_str})"
