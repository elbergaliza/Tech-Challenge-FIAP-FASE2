
# # usa os defaults (populacao=100, geracoes=50, mutacao=0.1, torneio=3)
# python -m ag.ag_RandomForest

# # sobrescreve qualquer combinação de parâmetros
# python -m ag.ag_RandomForest --populacao 200 --geracoes 100 --mutacao 0.2 --torneio 5

import argparse
import threading

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from ag.carga import carregar_modelo_completo, carregar_dataframe, carregar_split
from ag.classes.populacao import Populacao
from ag.salvar_modelo_final import treinar_e_salvar_modelo_final

# =============================================================================
# Parâmetros do Algoritmo Genético (via linha de comando ou defaults)
# =============================================================================
parser = argparse.ArgumentParser(
    description="Algoritmo Genético para otimização de hiperparâmetros do RandomForestClassifier",
)
parser.add_argument(
    "--populacao", type=int, default=100,
    help="Tamanho da população (default: 100)",
)
parser.add_argument(
    "--geracoes", type=int, default=50,
    help="Número máximo de gerações (default: 50)",
)
parser.add_argument(
    "--mutacao", type=float, default=0.1,
    help="Taxa de mutação por gene, entre 0.0 e 1.0 (default: 0.1)",
)
parser.add_argument(
    "--torneio", type=int, default=3,
    help="Tamanho do torneio para seleção (default: 3)",
)
args = parser.parse_args()

POPULATION_SIZE: int = args.populacao
MAX_GENERATIONS: int = args.geracoes
TAXA_MUTACAO: float = args.mutacao
TAMANHO_TORNEIO: int = args.torneio


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
best_params = melhor.hiperparametros
treinar_e_salvar_modelo_final(
    best_params=best_params,
    split=split,
    output_path="data/modelo_completo.joblib",
    target_name="HOSPITALIZ",
)
print("[OK] Saved final optimized model to data/modelo_completo.joblib")
print("\n--- Resultado Final ---")
print(melhor)

# =========================================================================
# Comparação de hiperparâmetros: modelo carregado vs melhor indivíduo do AG
# =========================================================================
hp_modelo = model.hiperparametros
hp_melhor = melhor.hiperparametros
todas_chaves = sorted(set(hp_modelo) | set(hp_melhor))

df_hp = pd.DataFrame({
    "Hiperparâmetro": todas_chaves,
    "Modelo carregado": [hp_modelo.get(k, "—") for k in todas_chaves],
    "Melhor AG": [hp_melhor.get(k, "—") for k in todas_chaves],
})
df_hp["Alterado?"] = df_hp.apply(
    lambda r: "Sim" if r["Modelo carregado"] != r["Melhor AG"] else "", axis=1,
)

print("\n=== Comparação de Hiperparâmetros ===")
print(df_hp.to_string(index=False))

# =========================================================================
# Comparação de métricas de aptidão: modelo carregado vs melhor indivíduo
# =========================================================================
apt_modelo = model.aptidao
apt_melhor = melhor.aptidao
assert apt_melhor is not None, "Aptidão do melhor indivíduo não foi calculada."

metricas = {
    "Acurácia treino": (apt_modelo.acuracia_treino, apt_melhor.acuracia_treino),
    "Acurácia teste": (apt_modelo.acuracia_teste, apt_melhor.acuracia_teste),
    "ROC-AUC": (apt_modelo.roc_auc, apt_melhor.roc_auc),
    "Accuracy (report)": (apt_modelo.accuracy, apt_melhor.accuracy),
}

nomes = list(metricas.keys())
vals_modelo = [v[0] for v in metricas.values()]
vals_melhor = [v[1] for v in metricas.values()]
deltas = [m - c for c, m in metricas.values()]

df_apt = pd.DataFrame({
    "Métrica": nomes,
    "Modelo carregado": [f"{v:.4f}" for v in vals_modelo],
    "Melhor AG": [f"{v:.4f}" for v in vals_melhor],
    "Delta": [f"{d:+.4f}" for d in deltas],
})

print("\n=== Comparação de Métricas de Aptidão ===")
print(df_apt.to_string(index=False))
