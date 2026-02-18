"""
Ponto de entrada para carregar artefatos da pasta data/.

Utiliza as classes definidas em ag.modelos.
Pode ser executado individualmente para testar a carga dos arquivos.

Arquivos esperados na pasta data/:
  - modelo_completo.joblib
  - DENGBR25_processado.csv
  - dados_split.joblib
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Garante que a raiz do projeto está no path (para execução direta do script)
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from ag.modelos import CarregadorArtefatos, DadosSplit, DatasetProcessado, PacoteModelo

# Diretório data/ relativo à raiz do projeto
_DIR_DATA = _RAIZ / "data"


def carregar_modelo_completo(diretorio_data: Optional[Path] = None) -> PacoteModelo:
    """Carrega o pacote completo (modelo + metadata + scaler)."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_modelo_completo()


def carregar_dataframe(diretorio_data: Optional[Path] = None) -> DatasetProcessado:
    """Carrega o DataFrame processado (CSV)."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_dataframe()


def carregar_split(diretorio_data: Optional[Path] = None) -> DadosSplit:
    """Carrega o split treino/teste (X_train, X_test, y_train, y_test)."""
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_split()


def carregar_tudo(
    diretorio_data: Optional[Path] = None,
) -> Tuple[PacoteModelo, DatasetProcessado, DadosSplit]:
    """
    Carrega todos os artefatos.

    Returns:
        Tupla (PacoteModelo, DatasetProcessado, DadosSplit).
    """
    carregador = CarregadorArtefatos(diretorio_data=diretorio_data or _DIR_DATA)
    return carregador.carregar_tudo()


if __name__ == "__main__":
    print("Carregando arquivos da pasta data/...")
    print(f"Diretório base: {_DIR_DATA}\n")

    # 1. Pacote completo (modelo + metadata + scaler)
    pacote = carregar_modelo_completo()
    print("--- modelo_completo.joblib ---")
    print(pacote)
    print(f"Data treinamento: {pacote.data_treinamento}")
    print(f"Acurácia treino: {pacote.acuracia_treino:.4f}")
    print(f"Acurácia teste: {pacote.acuracia_teste:.4f}")
    print(f"ROC AUC: {pacote.roc_auc:.4f}")
    print("Classification report:")
    print(f"  accuracy: {pacote.accuracy:.4f}")
    for classe, metricas in pacote.classification_report.items():
        # Ignora entradas do sklearn que não são classes (accuracy, macro avg, weighted avg)
        if not isinstance(metricas, dict) or "precision" not in metricas:
            continue
        support = (
            int(metricas["support"])
            if isinstance(metricas["support"], float)
            else metricas["support"]
        )
        print(
            f"  {classe}: precision={metricas['precision']:.4f}  "
            f"recall={metricas['recall']:.4f}  f1-score={metricas['f1-score']:.4f}  "
            f"support={support}"
        )
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
