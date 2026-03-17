from .generator_base import LaudoGenerator
from .laudos_type import EntradaLaudo


class TemplateLaudoGenerator(LaudoGenerator):
    def gerar(self, entrada: EntradaLaudo) -> str:
        r = entrada.resultado
        c = entrada.contexto

        proba = r.probabilidade_positiva
        if proba is None:
            risk_band = "undetermined"
        elif proba < 0.33:
            risk_band = "low"
        elif proba < 0.66:
            risk_band = "moderate"
        else:
            risk_band = "high"

        summary_items = "\n".join([f"- {k}: {v}" for k, v in entrada.resumo_exame.items()])

        roc_auc_txt = f"{c.roc_auc_global:.3f}" if c.roc_auc_global is not None else "not provided"
        acc_txt = f"{c.acuracia_teste:.3f}" if c.acuracia_teste is not None else "not provided"

        proba_txt = f"{proba:.4f}" if proba is not None else "not provided"

        return f"""\
1) Model output
- Model: {c.nome_modelo}
- Target: {c.target_name}
- Predicted class: {r.classe_predita}
- Probability (positive class): {proba_txt}
- Decision threshold: {r.limiar_decisao}

2) Interpretation
- The estimated probability is {risk_band} within the model scale.
- This result should be interpreted together with clinical assessment and other tests.

3) Points of attention
Summary of the provided data:
{summary_items if summary_items else "- (no data provided)"}

4) Limitations
- This text does not replace medical evaluation and does not constitute a definitive diagnosis."""