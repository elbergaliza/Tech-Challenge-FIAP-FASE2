"""
Representa a população de indivíduos do Algoritmo Genético (AG).
"""

import random
import time
from typing import Any, Dict, Optional, cast

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from ag.modelos.aptidao_modelo import AptidaoModelo
from ag.modelos.dados_split import DadosSplit
from ag.modelos.dataset_processado import DatasetProcessado
from ag.modelos.individuo import Individuo


class Populacao:
    """
    Representa a população de indivíduos do Algoritmo Genético.
    Contém os métodos aplicados à população: geração inicial, avaliação de
    aptidão, seleção, cruzamento, mutação e substituição.
    """

    def __init__(self, individuos: list[Individuo]) -> None:
        """
        Inicializa a população.

        Args:
            individuos: Lista de indivíduos da população.
        """
        self._individuos = list(individuos)

    @property
    def individuos(self) -> list[Individuo]:
        """Lista de indivíduos da população."""
        return self._individuos

    @property
    def tamanho(self) -> int:
        """Tamanho da população."""
        return len(self._individuos)

    # -------------------------------------------------------------------------
    # Métodos de fábrica
    # -------------------------------------------------------------------------

    @classmethod
    def gerar_inicial(
        cls,
        tamanho: int,
        split: Optional[DadosSplit] = None,
        dataset: Optional[DatasetProcessado] = None,
        incluir_default: bool = True,
    ) -> "Populacao":
        """
        Gera a população inicial com indivíduos aleatórios.

        Opcionalmente inclui o indivíduo default (solução de referência) como
        primeiro elemento para garantir diversidade e baseline.

        Args:
            tamanho: Número de indivíduos na população.
            split: Dados de treino/teste (opcional, usado em avaliação posterior).
            dataset: Dataset processado (opcional).
            incluir_default: Se True, inclui Individuo.default() como primeiro
                indivíduo. O restante é preenchido com indivíduos aleatórios.

        Returns:
            Nova Populacao com tamanho especificado.
        """
        individuos: list[Individuo] = []

        if incluir_default and tamanho > 0:
            individuos.append(Individuo.default())

        while len(individuos) < tamanho:
            individuos.append(Individuo.gerar_aleatorio())

        return cls(individuos)

    # -------------------------------------------------------------------------
    # Avaliação de aptidão
    # -------------------------------------------------------------------------

    def avaliar_aptidao(
        self,
        split: DadosSplit,
        metrica: str = "roc_auc",  # TODO: acrescentar as outras metricas de aptidao
    ) -> tuple[float, list[float]]:
        """
        Avalia a aptidão de todos os indivíduos da população.

        Treina um RandomForestClassifier com os hiperparâmetros de cada
        indivíduo e calcula a métrica de aptidão no conjunto de teste.

        Args:
            split: Dados de treino e teste (X_train, X_test, y_train, y_test).
            metrica: Métrica de avaliação. Opções: "roc_auc" (padrão).

        Returns:
            Tupla (tempo_total_segundos, lista de tempos por indivíduo).
        """
        tempos: list[float] = []
        tempo_total_inicio = time.perf_counter()

        for individuo in self._individuos:
            t_inicio = time.perf_counter()

            modelo = RandomForestClassifier(**individuo.hiperparametros)
            modelo.fit(split.X_train, split.y_train)

            y_pred_proba = modelo.predict_proba(split.X_test)
            y_test = split.y_test

            if isinstance(y_pred_proba, list):
                y_pred_proba_array = y_pred_proba[0]
            else:
                y_pred_proba_array = y_pred_proba

            if y_pred_proba_array.shape[1] == 2:
                y_score = y_pred_proba_array[:, 1]
            else:
                y_score = y_pred_proba_array
            # Probabilidades para ROC AUC; rótulos previstos para relatório
            y_pred = modelo.predict(split.X_test)

            try:
                roc_auc = float(roc_auc_score(
                    y_test, y_score, multi_class="ovr"))
            except ValueError:
                roc_auc = 0.0

            class_rep = classification_report(
                split.y_test, y_pred,
                labels=[0.0, 1.0],     # [np.float64(0.0), np.float64(1.0)]
                target_names=['Não grave', 'Grave'],   # ['Não grave', 'Grave']
                digits=4,
                zero_division="warn",  # "warn", "strict", "raise"
                output_dict=True)
            # output_dict=True guarantees a dict; sklearn stubs declare str | dict
            class_rep_dict: Dict[str, Any] = cast(Dict[str, Any], class_rep)

            acuracia_treino = modelo.score(split.X_train, split.y_train)
            acuracia_teste = modelo.score(split.X_test, split.y_test)

            individuo.aptidao = AptidaoModelo(
                roc_auc=roc_auc, classification_report=class_rep_dict, acuracia_treino=acuracia_treino, acuracia_teste=acuracia_teste)
            tempos.append(time.perf_counter() - t_inicio)

        tempo_total = time.perf_counter() - tempo_total_inicio
        return tempo_total, tempos

    def ordenar_por_aptidao(self) -> None:
        """
        Ordena a população por aptidão em ordem decrescente (melhor primeiro).
        """
        self._individuos.sort(
            key=lambda i: i.aptidao.roc_auc if i.aptidao is not None else 0.0,
            reverse=True,
        )

    # -------------------------------------------------------------------------
    # Seleção
    # -------------------------------------------------------------------------

    def melhor_individuo(self) -> Individuo:
        """
        Retorna o melhor indivíduo da população (maior aptidão).

        Returns:
            O melhor Individuo.

        Raises:
            ValueError: Se a população estiver vazia ou sem aptidão calculada.
        """
        if not self._individuos:
            raise ValueError("População vazia.")
        self.ordenar_por_aptidao()
        return self._individuos[0]

    def selecionar_torneio(self, tamanho_torneio: int = 3) -> Individuo:
        """
        Seleção por torneio: escolhe aleatoriamente `tamanho_torneio` indivíduos
        e retorna o de maior aptidão.

        Args:
            tamanho_torneio: Número de indivíduos no torneio.

        Returns:
            Indivíduo selecionado.
        """
        participantes = random.sample(
            self._individuos,
            min(tamanho_torneio, len(self._individuos)),
        )
        return max(
            participantes,
            key=lambda i: i.aptidao.roc_auc if i.aptidao is not None else 0.0,
        )

    # -------------------------------------------------------------------------
    # Nova geração (elitismo + seleção + cruzamento + mutação)
    # -------------------------------------------------------------------------

    def gerar_nova_geracao(
        self,
        tamanho: int,
        taxa_mutacao: float = 0.1,
        tamanho_torneio: int = 3,
    ) -> "Populacao":
        """
        Gera uma nova população aplicando elitismo, seleção, cruzamento e mutação.

        - Elitismo: o melhor indivíduo é preservado.
        - Seleção: torneio para escolher pais.
        - Cruzamento: crossover uniforme entre dois pais.
        - Mutação: aplicada aos descendentes.

        Args:
            tamanho: Tamanho da nova população.
            taxa_mutacao: Probabilidade de mutação por gene (0.0 a 1.0).
            tamanho_torneio: Tamanho do torneio para seleção.

        Returns:
            Nova Populacao (indivíduos sem aptidão calculada).
        """
        self.ordenar_por_aptidao()
        elite = self._individuos[0].copiar()
        nova_pop: list[Individuo] = [elite]

        while len(nova_pop) < tamanho:
            pai1 = self.selecionar_torneio(tamanho_torneio)
            pai2 = self.selecionar_torneio(tamanho_torneio)

            filho1, filho2 = pai1.cruzar(pai2)
            filho1 = filho1.mutar(taxa_mutacao)
            filho2 = filho2.mutar(taxa_mutacao)

            nova_pop.append(filho1)
            if len(nova_pop) < tamanho:
                nova_pop.append(filho2)

        return Populacao(nova_pop)

    # -------------------------------------------------------------------------
    # Utilitários
    # -------------------------------------------------------------------------

    def substituir(self, nova_populacao: "Populacao") -> None:
        """
        Substitui os indivíduos da população pelos da nova população.

        Args:
            nova_populacao: Populacao que substituirá a atual.
        """
        self._individuos = list(nova_populacao.individuos)

    def __repr__(self) -> str:
        return f"Populacao(tamanho={self.tamanho})"
