from ag.carga import carregar_modelo_completo, carregar_dataframe, carregar_split
from ag.classes.populacao import Populacao
import threading
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

# =============================================================================
# Parâmetros do Algoritmo Genético
# =============================================================================
# TODO: ajustar os parâmetros do AG
POPULATION_SIZE = 100
MAX_GENERATIONS = 50
TAXA_MUTACAO = 0.1
TAMANHO_TORNEIO = 3


def parar_ag(geracao_atual: int) -> bool:
    return geracao_atual >= MAX_GENERATIONS


# =============================================================================
# DASH
# =============================================================================
app = Dash(__name__)

# Histórico compartilhado AG → Dash
historico = {
    "geracao": [],
    "melhor_roc": [],
    "tempo_total": [],
    "tempo_medio": [],
    "cv_media": [],
    "cv_std": [],
    "hiperparametros": [],
}

app.layout = html.Div(
    [
        html.H2("Evolução do Algoritmo Genético"),
        dcc.Graph(id="grafico-principal"),
        dcc.Graph(id="grafico-cv"),
        dcc.Graph(id="grafico-dist"),
        dcc.Graph(id="grafico-corr"),
        dcc.Interval(id="intervalo", interval=1000, n_intervals=0),
    ]
)


@app.callback(
    Output("grafico-principal", "figure"),
    Output("grafico-cv", "figure"),
    Output("grafico-dist", "figure"),
    Output("grafico-corr", "figure"),
    Input("intervalo", "n_intervals"),
)
def atualizar_dash(_):
    # =======================
    # Gráfico principal
    # =======================
    fig_main = go.Figure()

    fig_main.add_trace(
        go.Scatter(
            x=historico["geracao"],
            y=historico["melhor_roc"],
            mode="lines+markers",
            name="Melhor ROC-AUC",
        )
    )

    fig_main.add_trace(
        go.Scatter(
            x=historico["geracao"],
            y=historico["tempo_total"],
            mode="lines+markers",
            name="Tempo total",
            yaxis="y2",
        )
    )

    fig_main.update_layout(
        template="plotly_dark",
        xaxis_title="Geração",
        yaxis=dict(title="ROC-AUC"),
        yaxis2=dict(title="Tempo (s)", overlaying="y", side="right"),
    )

    # =======================
    # Gráfico CV
    # =======================
    fig_cv = go.Figure()

    fig_cv.add_trace(
        go.Scatter(
            x=historico["geracao"],
            y=historico["cv_media"],
            mode="lines+markers",
            name="CV média",
        )
    )

    fig_cv.add_trace(
        go.Scatter(
            x=historico["geracao"],
            y=historico["cv_std"],
            mode="lines+markers",
            name="CV std",
        )
    )

    fig_cv.update_layout(
        template="plotly_dark",
        title="Validação Cruzada do Melhor Indivíduo",
        xaxis_title="Geração",
        yaxis_title="Score",
    )

    # =======================
    # Distribuição hiperparâmetros
    # =======================
    if historico["hiperparametros"]:
        df_hp = pd.DataFrame(historico["hiperparametros"])
        fig_dist = px.box(df_hp, template="plotly_dark",
                          title="Distribuição dos Hiperparâmetros")

        corr = df_hp.corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=True, template="plotly_dark",
                             title="Correlação entre Hiperparâmetros")
    else:
        fig_dist = go.Figure()
        fig_corr = go.Figure()

    return fig_main, fig_cv, fig_dist, fig_corr


# =============================================================================
# Realiza a carga do modelo, dos dados e do split
# - Modelo/dataset são carregados para validar os artefatos e metadados
# - O AG usa somente o split (X/y) para treinar e avaliar indivíduos
# =============================================================================
model = carregar_modelo_completo()
dataset = carregar_dataframe()
split = carregar_split()

# =============================================================================
# FLUXOGRAMA AG - INÍCIO → Gera População Inicial
# Um indivíduo = uma possível solução.
# TECNICA: Geração Aleatória
# =============================================================================
population = Populacao.gerar_inicial(
    tamanho=POPULATION_SIZE,
    split=split,
    dataset=dataset,
    incluir_default=True,
)

# =============================================================================
# FLUXOGRAMA AG - Main loop (ciclo iterativo)
# Condição de término: Número máximo de gerações atingido.
# TODO: colocar parametro para selecionar a condicao de termino:
# - Número máximo de gerações atingido.
# - Convergência da aptidão.
# - Tempo limite de execução.
# - Avaliação da aptidão otima.
# =============================================================================
geracao = 0


# =============================================================================
# Thread Dash
# =============================================================================
def iniciar_dashboard():
    app.run(debug=False, use_reloader=False)


threading.Thread(target=iniciar_dashboard, daemon=True).start()

# =============================================================================
# LOOP AG
# =============================================================================
while not parar_ag(geracao):
    # =========================================================================
    # FLUXOGRAMA AG - Avalia Aptidão dos Indivíduos
    # Selecionamos as melhores soluções usando função de aptidão (ROC-AUC). # TODO: retirar comentario de avaliar aptidao
    # =========================================================================
    tempo_total, tempos_individuos = population.avaliar_aptidao(split=split)
    tempo_medio_ind = sum(tempos_individuos) / \
        len(tempos_individuos) if tempos_individuos else 0

    # TODO: colocar as duas linhas abaixo dentro de melhor individuo para usar o parametro de aptidao
    melhor = population.melhor_individuo()
    melhor_roc_auc = melhor.aptidao.roc_auc if melhor.aptidao else 0.0

    # CV do melhor indivíduo (assumindo que aptidão guarda isso)
    melhor_cv = getattr(melhor.aptidao, "cv_media", melhor_roc_auc)
    melhor_std = getattr(melhor.aptidao, "cv_std", 0.0)

    print(f"\n--- Melhor individuo ---")
    print(melhor)

    # Histórico
    historico["geracao"].append(geracao)
    historico["melhor_roc"].append(melhor_roc_auc)
    historico["tempo_total"].append(tempo_total)
    historico["tempo_medio"].append(tempo_medio_ind)
    historico["cv_media"].append(melhor_cv)
    historico["cv_std"].append(melhor_std)

    # salva hiperparâmetros da população inteira
    for ind in population.individuos:
        historico["hiperparametros"].append(ind.hiperparametros)

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
