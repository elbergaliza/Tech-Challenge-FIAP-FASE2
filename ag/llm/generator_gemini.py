from .client_base import LLMClient
from .generator_base import LaudoGenerator
from .laudos_type import EntradaLaudo
from .prompts import SYSTEM_PROMPT, FORMAT_INSTRUCTIONS


class GeminiLaudoGenerator(LaudoGenerator):
    def __init__(self, model: str, client: LLMClient, max_output_tokens: int = 450):
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def _build_user_prompt(self, entrada: EntradaLaudo) -> str:
        r = entrada.resultado
        c = entrada.contexto

        exam_lines = "\n".join([f"- {k}: {v}" for k, v in entrada.resumo_exame.items()])

        return (
            f"{FORMAT_INSTRUCTIONS}\n\n"
            f"Available data:\n"
            f"- Model: {c.nome_modelo}\n"
            f"- Target: {c.target_name}\n"
            f"- Predicted class: {r.classe_predita}\n"
            f"- Positive class probability: {r.probabilidade_positiva}\n"
            f"- Decision threshold: {r.limiar_decisao}\n"
            f"- Global ROC-AUC (if available): {c.roc_auc_global}\n"
            f"- Test accuracy (if available): {c.acuracia_teste}\n"
            f"- Exam summary:\n{exam_lines if exam_lines else '(no data provided)'}\n"
            f"- Clinical text (optional): {entrada.texto_clinico or '(none)'}\n"
        )

    def gerar(self, entrada: EntradaLaudo) -> str:
        return self.client.generate(
            self.model,
            self._build_user_prompt(entrada),
            system_prompt=SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
            temperature=0.2,
        )
