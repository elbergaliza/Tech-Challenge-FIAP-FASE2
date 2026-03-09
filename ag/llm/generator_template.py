from .generator_base import LaudoGenerator
from .laudos_type import EntradaLaudo


class TemplateLaudoGenerator(LaudoGenerator):
    def gerar(self, entrada: EntradaLaudo) -> str:
        r = entrada.resultado
        c = entrada.contexto

        proba = r.probabilidade_positiva
        if proba is None:
            risk_band = "indeterminado"
        elif proba < 0.33:
            risk_band = "baixo"
        elif proba < 0.66:
            risk_band = "moderado"
        else:
            risk_band = "alto"

        summary_items = "\n".join([f"- {k}: {v}" for k, v in entrada.resumo_exame.items()])

        roc_auc_txt = f"{c.roc_auc_global:.3f}" if c.roc_auc_global is not None else "não informado"
        acc_txt = f"{c.acuracia_teste:.3f}" if c.acuracia_teste is not None else "não informado"

        proba_txt = f"{proba:.4f}" if proba is not None else "não informado"

        return f"""\
1) Resultado do modelo
- Modelo: {c.nome_modelo}
- Alvo: {c.target_name}
- Classe predita: {r.classe_predita}
- Probabilidade (classe positiva): {proba_txt}
- Limiar de decisão: {r.limiar_decisao}

2) Interpretação
- A probabilidade estimada está em nível {risk_band} na escala do modelo.
- Este resultado deve ser interpretado em conjunto com avaliação clínica e outros exames.

3) Pontos de atenção
Resumo dos dados fornecidos:
{summary_items if summary_items else "- (sem dados fornecidos)"}

4) Limitações
- Métricas globais de referência: ROC-AUC={roc_auc_txt}; acurácia de teste={acc_txt}.
- O modelo pode errar (falsos positivos/falsos negativos).
- Este resultado não substitui avaliação médica.
- Este texto não constitui diagnóstico definitivo.
"""
