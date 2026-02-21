# Tech-Challenge-FIAP-FASE2
Solução do desafio da Fase 2 do grupo 57 (FIAP).

## Visão geral
Este projeto executa um Algoritmo Genético (AG) para otimizar hiperparâmetros
de um `RandomForestClassifier`, utilizando artefatos pré-processados
armazenados em `data/`.

## Estrutura do projeto
- `ag/`: pacote principal do projeto.
- `ag/ag_RandomForest.py`: script principal que executa o AG.
- `ag/carregar_dados.py`: utilitário para carregar/validar artefatos em `data/`.
- `ag/modelos/`: classes de domínio (população, indivíduo, métricas e artefatos).
- `data/`: artefatos de dados e modelo.
- `docs/`: documentação complementar (relatórios).

## Artefatos esperados em `data/`
Os nomes são obrigatórios para a carga automática:
- `modelo_completo.joblib` (dict com `modelo`, `aptidao`, `metadata` e `scaler`)
- `DENGBR25_processado.csv` (dataset processado)
- `dados_split.joblib` (dict com `X_train`, `X_test`, `y_train`, `y_test`)

## Requisitos
- Python `3.13` (ver `.python-version`)
- Pip

## Instalação
Execute os comandos na raiz do repositório.

### Windows (PowerShell)
```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux/macOS
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execução
### Validar artefatos (opcional)
```bash
python -m ag.carregar_dados
```

### Rodar o Algoritmo Genético
```bash
python -m ag.ag_RandomForest
```

## Notas técnicas
- O AG utiliza `dados_split.joblib` para treinar e avaliar indivíduos.
- O ROC AUC é calculado com probabilidades (`predict_proba`); o
  `classification_report` usa rótulos previstos (`predict`).
- Caso os artefatos não estejam em `data/`, passe outro diretório ao carregar
  via `ag.carregar_dados` (funções aceitam `diretorio_data`).
