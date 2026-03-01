"""
Carregador dos artefatos salvos (modelo_completo.joblib, CSV, dados_split.joblib).
"""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd

from ag.classes.aptidao_modelo import AptidaoModelo
from ag.carga.dados_split import DadosSplit
from ag.carga.dataset_processado import DatasetProcessado
from ag.carga.modelo import PacoteModelo


class CarregadorArtefatos:
    """
    Carrega os artefatos a partir de um diretório (ex.: pasta data/).
    Retorna instâncias de PacoteModelo, DatasetProcessado e DadosSplit.
    """

    ARQUIVO_MODELO = "modelo_completo.joblib"
    ARQUIVO_CSV = "DENGBR25_processado.csv"
    ARQUIVO_SPLIT = "dados_split.joblib"

    def __init__(self, diretorio_data: Optional[Path] = None) -> None:
        """
        Args:
            diretorio_data: Pasta onde estão os arquivos. Se None, usa pasta
                'data' relativa ao diretório do projeto (raiz do repo).
        """
        if diretorio_data is not None:
            self._dir = Path(diretorio_data).resolve()
        else:
            # Projeto: ag/ e data/ na mesma raiz
            self._dir = Path(__file__).resolve().parent.parent.parent / "data"
        self._dir = self._dir

    @property
    def diretorio(self) -> Path:
        """Diretório configurado para leitura dos artefatos."""
        return self._dir

    def carregar_modelo_completo(self) -> PacoteModelo:
        """Carrega modelo_completo.joblib e retorna um PacoteModelo."""
        path = self._dir / self.ARQUIVO_MODELO
        # Dict com chaves modelo, aptidao, metadata e scaler
        # pacote_completo = {
        #     # Modelo treinado
        #     'modelo': rf_convertido,

        #     #Metricas
        #     'aptidao': {
        #     'acuracia_treino': acuracia_treino,
        #     'acuracia_teste': acuracia_teste,
        #     'roc_auc': roc_auc,
        #     'classification_report': class_rep
        #     },

        #     # Metadados
        #     'metadata': {
        #         'data_treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #         'sklearn_version': "1.6.1",
        #         'pandas_version': pd.__version__,
        #         'target_name': 'HOSPITALIZ',
        #         'feature_names': feature_names_convert,
        #         'hiperparametros': {
        #         'n_estimators': 50,
        #         'max_depth': 5,
        #         'random_state': 42,
        #         'min_samples_leaf': 10,
        #         'min_samples_split': 10,
        #         'max_features': "sqrt",
        #         'n_jobs': -1
        #         }
        #     },

        #     # Preprocessadores (se houver)
        #     'scaler': scaler if 'scaler' in locals() else None,
        # }
        pacote = joblib.load(path)
        return PacoteModelo(
            modelo=pacote["modelo"],
            aptidao=AptidaoModelo.from_dict(pacote["aptidao"]),
            metadata=pacote["metadata"],
            scaler=pacote.get("scaler"),
        )

    def carregar_dataframe(self) -> DatasetProcessado:
        """Carrega o CSV processado e retorna um DatasetProcessado."""
        path = self._dir / self.ARQUIVO_CSV
        # CSV processado já deve conter as features finais e o target
        df = pd.read_csv(path)
        return DatasetProcessado(df=df, caminho=path)

    def carregar_split(self) -> DadosSplit:
        """Carrega dados_split.joblib e retorna um DadosSplit."""
        path = self._dir / self.ARQUIVO_SPLIT
        # Dict com X_train, X_test, y_train, y_test
        # dados_split = {
        #     'X_train': X_train_convert,
        #     'X_test': X_test_convert,
        #     'y_train': y_train_convert,
        #     'y_test': y_test_convert
        # }
        dados = joblib.load(path)
        return DadosSplit(
            X_train=dados["X_train"],
            X_test=dados["X_test"],
            y_train=dados["y_train"],
            y_test=dados["y_test"],
        )

    def carregar_tudo(
        self,
    ) -> Tuple[PacoteModelo, DatasetProcessado, DadosSplit]:
        """
        Carrega todos os artefatos.
        Retorna (pacote_modelo, dataset_processado, dados_split).
        """
        pacote = self.carregar_modelo_completo()
        dataset = self.carregar_dataframe()
        split = self.carregar_split()
        return pacote, dataset, split
