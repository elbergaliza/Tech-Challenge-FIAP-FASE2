#!/usr/bin/env python3
"""Estimativa simples de custo para execuções de HPO no SageMaker.

Fórmula:
  custo_execucao = preco_hora_instancia * horas_por_training_job * max_jobs
  custo_total = custo_execucao * numero_execucoes

Observação: é estimativa conservadora, sem descontos/free tier e sem overhead.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimador de custo do fluxo HPO")
    parser.add_argument("--hourly", type=float, default=0.23, help="Preço/hora da instância de treino")
    parser.add_argument("--hours-per-job", type=float, default=0.25, help="Horas médias por training job")
    parser.add_argument("--max-jobs", type=int, default=12, help="Quantidade de jobs por execução de HPO")
    args = parser.parse_args()

    cost_per_execution = args.hourly * args.hours_per_job * args.max_jobs

    print("Assumptions")
    print(f"- hourly: ${args.hourly:.4f}/h")
    print(f"- hours-per-job: {args.hours_per_job:.2f}h")
    print(f"- max-jobs: {args.max_jobs}")
    print(f"- cost-per-execution: ${cost_per_execution:.2f}")

    for runs in (1, 100, 1000):
        total = cost_per_execution * runs
        print(f"runs={runs}: ${total:,.2f}")


if __name__ == "__main__":
    main()
