from ag.carregar_dados import carregar_modelo_completo, carregar_dataframe, carregar_split
from ag.modelos.populacao import Populacao

# =============================================================================
# Parâmetros do Algoritmo Genético
# =============================================================================
POPULATION_SIZE = 100
MAX_GENERATIONS = 50
TAXA_MUTACAO = 0.1
TAMANHO_TORNEIO = 3


def parar_ag(geracao_atual: int) -> bool:
    """
    Verifica se o algoritmo deve parar.

    Condições de parada:
    - Número máximo de gerações atingido.

    Args:
        geracao_atual: Número da geração atual (0-based).

    Returns:
        True se o AG deve parar, False caso contrário.
    """
    return geracao_atual >= MAX_GENERATIONS


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
population = Populacao.gerar_inicial(
    tamanho=POPULATION_SIZE,
    split=split,
    dataset=dataset,
    incluir_default=True,
)

# =============================================================================
# FLUXOGRAMA AG - Main loop (ciclo iterativo)
# Verifica condição de término: running=False (tecla Q ou fechar janela) → FIM
# =============================================================================
geracao = 0

while not parar_ag(geracao):
    # =========================================================================
    # FLUXOGRAMA AG - Avalia Aptidão dos Indivíduos
    # Selecionamos as melhores soluções usando função de aptidão (ROC-AUC).
    # =========================================================================
    tempo_total, tempos_individuos = population.avaliar_aptidao(split=split)
    tempo_medio_ind = sum(tempos_individuos) / len(tempos_individuos) if tempos_individuos else 0
    print(
        f"Geração {geracao + 1}/{MAX_GENERATIONS} | "
        f"Melhor aptidão: {population.melhor_individuo().aptidao:.4f} | "
        f"Tempo total: {tempo_total:.2f}s | "
        f"Tempo médio/ind: {tempo_medio_ind:.2f}s"
    )

    # =========================================================================
    # FLUXOGRAMA AG - ETAPA: Seleção + Cruzamento + Mutação
    # Inicializa a nova população com o melhor indivíduo (elitismo)
    # =========================================================================
    nova_population = population.gerar_nova_geracao(
        tamanho=POPULATION_SIZE,
        taxa_mutacao=TAXA_MUTACAO,
        tamanho_torneio=TAMANHO_TORNEIO,
    )

    # =========================================================================
    # FLUXOGRAMA AG - ETAPA: Substitui População Antiga
    # Nova geração → retorna a "Avalia Aptidão"
    # =========================================================================
    population.substituir(nova_population)
    geracao += 1

# =============================================================================
# Resultado final
# =============================================================================
population.avaliar_aptidao(split=split)
melhor = population.melhor_individuo()
print(f"\n--- Resultado Final ---")
print(f"Melhor indivíduo: {melhor}")
print(f"Hiperparâmetros: {melhor.hiperparametros}")
