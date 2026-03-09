#!/usr/bin/env python3
"""Cria e/ou coleta resultados de HPO no SageMaker.

Fluxos:
- Criacao: informar --role-arn e --bucket (opcionalmente --wait para aguardar e coletar).
- Coleta: informar apenas --base-job-name para aguardar/coletar job existente.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria e/ou coleta resultados de HPO no SageMaker")
    parser.add_argument("--base-job-name", required=True, help="Nome do HyperParameterTuningJob")
    parser.add_argument("--region", default="us-east-1", help="Região AWS")
    parser.add_argument("--role-arn", help="IAM role de execução do SageMaker")
    parser.add_argument("--bucket", help="Bucket S3 com inputs")
    parser.add_argument("--prefix", default="tech-challenge-hpo", help="Prefixo S3")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--max-parallel-jobs", type=int, default=2)
    parser.add_argument("--max-runtime-seconds", type=int, default=3600)
    parser.add_argument("--target", default="HOSPITALIZ")
    parser.add_argument(
        "--training-image-uri",
        default="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    )
    parser.add_argument(
        "--source-code-s3-uri",
        default="",
        help="S3 URI do tar.gz contendo train_rf.py (sagemaker-source.tar.gz).",
    )
    parser.add_argument("--objective-metric-name", default="validation:roc_auc")
    parser.add_argument("--objective-metric-regex", default=r"validation:roc_auc=([0-9\.]+)")
    parser.add_argument("--wait", action="store_true", help="Aguardar conclusão do HPO")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Intervalo de polling do HPO")
    parser.add_argument("--results-dir", default="data", help="Pasta local para salvar relatório")
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    tuning_job_name = args.base_job_name

    has_create_args = bool(args.role_arn and args.bucket)
    if has_create_args:
        _create_hpo(sm, args, tuning_job_name)
        print(f"HPO iniciado: {tuning_job_name}")
    elif args.role_arn or args.bucket:
        raise ValueError("Para criar HPO informe ambos: --role-arn e --bucket")

    if args.wait or not has_create_args:
        _wait_and_collect(sm, args, tuning_job_name)


def _create_hpo(sm, args, tuning_job_name: str) -> None:
    source_code_s3_uri = args.source_code_s3_uri or f"s3://{args.bucket}/{args.prefix}/code/sagemaker-source.tar.gz"
    train_input = f"s3://{args.bucket}/{args.prefix}/input/train.csv"
    val_input = f"s3://{args.bucket}/{args.prefix}/input/validation.csv"
    output_path = f"s3://{args.bucket}/{args.prefix}/models"

    sm.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name,
        HyperParameterTuningJobConfig={
            "Strategy": "Bayesian",
            "HyperParameterTuningJobObjective": {
                "Type": "Maximize",
                "MetricName": args.objective_metric_name,
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": args.max_jobs,
                "MaxParallelTrainingJobs": args.max_parallel_jobs,
            },
            "ParameterRanges": {
                "IntegerParameterRanges": [
                    {"Name": "n-estimators", "MinValue": "20", "MaxValue": "200", "ScalingType": "Auto"},
                    {"Name": "max-depth", "MinValue": "3", "MaxValue": "20", "ScalingType": "Auto"},
                    {"Name": "min-samples-split", "MinValue": "2", "MaxValue": "60", "ScalingType": "Auto"},
                    {"Name": "min-samples-leaf", "MinValue": "1", "MaxValue": "20", "ScalingType": "Auto"},
                ],
                "ContinuousParameterRanges": [
                    {"Name": "max-features", "MinValue": "0.3", "MaxValue": "1.0", "ScalingType": "Auto"}
                ],
            },
            "TrainingJobEarlyStoppingType": "Auto",
        },
        TrainingJobDefinition={
            "StaticHyperParameters": {
                "target": args.target,
                "random-state": "42",
                "n-jobs": "-1",
                "sagemaker_program": "train_rf.py",
                "sagemaker_submit_directory": source_code_s3_uri,
                "sagemaker_region": args.region,
            },
            "AlgorithmSpecification": {
                "TrainingImage": args.training_image_uri,
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {
                        "Name": args.objective_metric_name,
                        "Regex": args.objective_metric_regex,
                    }
                ],
            },
            "RoleArn": args.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "ContentType": "text/csv",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_input,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "validation",
                    "ContentType": "text/csv",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": val_input,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
            "OutputDataConfig": {"S3OutputPath": output_path},
            "ResourceConfig": {
                "InstanceType": args.instance_type,
                "InstanceCount": args.instance_count,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        },
    )


def _wait_and_collect(sm, args, tuning_job_name: str) -> None:
    terminal_states = {"Completed", "Failed", "Stopped"}
    while True:
        resp = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)
        status = resp["HyperParameterTuningJobStatus"]
        print(f"status={status}")
        if status in terminal_states:
            if status != "Completed":
                raise RuntimeError(f"HPO terminou com status {status}")
            best_job_info = resp.get("BestTrainingJob", {})
            best_job_name = best_job_info.get("TrainingJobName", "")
            print(f"HPO concluído. Best training job: {best_job_name}")
            _collect_results(sm, args, tuning_job_name, best_job_name, resp)
            return
        time.sleep(args.poll_seconds)


def _collect_results(sm, args, tuning_job_name: str, best_job_name: str, hpo_resp: dict) -> None:
    best_job_info = hpo_resp.get("BestTrainingJob", {})

    best_metrics: dict = {}
    objective = best_job_info.get("FinalHyperParameterTuningJobObjectiveMetric", {})
    best_metrics[objective.get("MetricName", args.objective_metric_name)] = objective.get("Value")

    if best_job_name:
        tj = sm.describe_training_job(TrainingJobName=best_job_name)
        for metric in tj.get("FinalMetricDataList", []):
            best_metrics[metric["MetricName"]] = metric["Value"]

    all_jobs = []
    objective_metric_name = args.objective_metric_name
    paginator = sm.get_paginator("list_training_jobs_for_hyper_parameter_tuning_job")
    for page in paginator.paginate(HyperParameterTuningJobName=tuning_job_name):
        for job in page.get("TrainingJobSummaries", []):
            objective_metric = job.get("FinalHyperParameterTuningJobObjectiveMetric", {})
            objective_value = objective_metric.get("Value")
            all_jobs.append(
                {
                    "job_name": job.get("TrainingJobName"),
                    "status": job.get("TrainingJobStatus"),
                    "hyperparameters": job.get("TunedHyperParameters", {}),
                    objective_metric_name: objective_value,
                }
            )
            print(
                f"job={job.get('TrainingJobName')} "
                f"status={job.get('TrainingJobStatus')} "
                f"{objective_metric_name}={objective_value}"
            )

    report = {
        "origem": "sagemaker_hpo",
        "hpo_job_name": tuning_job_name,
        "melhor_training_job": best_job_name,
        "hiperparametros": best_job_info.get("TunedHyperParameters", {}),
        "aptidao": best_metrics,
        "todos_jobs": all_jobs,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_name = tuning_job_name.replace("/", "-")
    out_path = results_dir / f"hpo_results_{safe_name}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path = results_dir / "report.json"
    latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Relatório salvo: {out_path}")
    print(f"Relatório atual (fixo): {latest_path}")


if __name__ == "__main__":
    main()
