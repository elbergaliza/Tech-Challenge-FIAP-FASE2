from ag.carregar import carregar_modelo_completo, carregar_dataframe, carregar_split
from ag.algoritmo_genetico import gerar_populacao_inicial


def parar_ag() -> bool:
    """
    Verifica se o algoritmo deve parar.
    """

    return False  # TODO: implementar a condicao de parada do algoritmo


# =============================================================================
# Realiza a carga do modelo, dos dados e do split
# =============================================================================
model = carregar_modelo_completo()
dataset = carregar_dataframe()
split = carregar_split()

# =============================================================================
# FLUXOGRAMA AG - INÍCIO → Gera População Inicial
# Um indivíduo = uma possível solução.
# =============================================================================
population = gerar_populacao_inicial(dataset=dataset, split=split)
POPULATION_SIZE = 100


# =============================================================================
# FLUXOGRAMA AG - Main loop (ciclo iterativo)
# Verifica condição de término: running=False (tecla Q ou fechar janela) → FIM
# =============================================================================

while not parar_ag():

    # =============================================================================
    # FLUXOGRAMA AG - Avalia Aptidão dos Indivíduos
    # Selecionamos as melhores soluções usando função de aptidão (menor distância = melhor).
    # Cada posição de population_fitness = distância total do caminho (indivíduo).
    # =============================================================================

    # TODO: adicionar o custo computacional, ou seja o tempo de execucao do algoritmo para cada geracao.

    # =============================================================================
    # FLUXOGRAMA AG - ETAPA: Seleção (etapa 1)
    # Inicializa a nova população com o melhor indivíduo (elitismo)
    # =============================================================================
    new_population = [population[0]]  # Keep the best individual: ELITISM

    # Repete até preencher a população com o tamanho desejado
    while len(new_population) < POPULATION_SIZE:

        # =============================================================================
        # FLUXOGRAMA AG - ETAPA: Seleção
        # =============================================================================

        # =============================================================================
        # FLUXOGRAMA AG - ETAPA: Cruzamento (Crossover). Cria nova solução a partir dos pais
        # =============================================================================

        # =============================================================================
        # FLUXOGRAMA AG - Mutação: introduz variações aleatórias no descendente
        # =============================================================================

        # =============================================================================
        # FLUXOGRAMA AG - ETAPA: Substitui População Antiga: nova geração → retorna a "Avalia Aptidão"
        # =============================================================================
        pass  # TODO: implementar a substituicao da populacao antiga
