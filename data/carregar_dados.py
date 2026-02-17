"""
Script para carregar os arquivos gerados no Google Colab (modelo, CSV e split).
Utiliza as classes definidas em ag.modelos.

Arquivos esperados na pasta data/:
  - modelo_completo.joblib
  - DENGBR25_processado.csv
  - dados_split.joblib
"""

import sys
from pathlib import Path

# Garante que a raiz do projeto está no path
_raiz = Path(__file__).resolve().parent.parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

from ag.modelos import CarregadorArtefatos, DadosSplit, DatasetProcessado, PacoteModelo


def carregar_modelo_completo() -> PacoteModelo:
    """Carrega o pacote completo (modelo + metadata + scaler)."""
    carregador = CarregadorArtefatos(diretorio_data=_raiz / "data")
    return carregador.carregar_modelo_completo()


def carregar_dataframe() -> DatasetProcessado:
    """Carrega o DataFrame processado (CSV)."""
    carregador = CarregadorArtefatos(diretorio_data=_raiz / "data")
    return carregador.carregar_dataframe()


def carregar_split() -> DadosSplit:
    """Carrega o split treino/teste (X_train, X_test, y_train, y_test)."""
    carregador = CarregadorArtefatos(diretorio_data=_raiz / "data")
    return carregador.carregar_split()


def carregar_tudo() -> tuple:
    """
    Carrega todos os artefatos.
    Retorna (PacoteModelo, DatasetProcessado, DadosSplit).
    """
    carregador = CarregadorArtefatos(diretorio_data=_raiz / "data")
    return carregador.carregar_tudo()


if __name__ == "__main__":
    print("Carregando arquivos da pasta data/...")
    print(f"Diretório base: {_raiz / 'data'}\n")

    # 1. Pacote completo (modelo + metadata + scaler)
    pacote = carregar_modelo_completo()
    print("--- modelo_completo.joblib ---")
    print(pacote)
    print(f"Data treinamento: {pacote.data_treinamento}")
    print(f"Acurácia treino: {pacote.acuracia_treino:.4f}")
    print(f"Acurácia teste: {pacote.acuracia_teste:.4f}")
    print(f"ROC AUC: {pacote.roc_auc:.4f}")
    print(f"Target: {pacote.target_name}")
    print(f"Scaler presente: {pacote.tem_scaler()}\n")

    # 2. DataFrame processado
    dataset = carregar_dataframe()
    print("--- DENGBR25_processado.csv ---")
    print(dataset)
    print(f"Colunas (primeiras 5): {dataset.colunas[:5]}...\n")

    # 3. Split treino/teste
    split = carregar_split()
    print("--- dados_split.joblib ---")
    print(split)
    print(f"  X_train: shape {split.shape_treino()}")
    print(f"  X_test: shape {split.shape_teste()}")

    print("\nCarregamento concluído.")
