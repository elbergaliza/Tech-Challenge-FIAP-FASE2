#!/usr/bin/env python3
"""Inicia HPO do RandomForest no SageMaker via boto3."""

from __future__ import annotations

import argparse
import time

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispara HPO de RandomForest no SageMaker")
    parser.add_argument("--role-arn", required=True, help="IAM role de execução do SageMaker")
    parser.add_argument("--bucket", required=True, help="Bucket S3 com inputs")
    parser.add_argument("--prefix", default="tech-challenge-hpo", help="Prefixo S3")
    parser.add_argument("--region", default="us-east-1", help="Região AWS")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--max-parallel-jobs", type=int, default=2)
    parser.add_argument("--max-runtime-seconds", type=int, default=3600)
    parser.add_argument("--base-job-name", default="rf-hpo")
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
    args = parser.parse_args()

    source_code_s3_uri = args.source_code_s3_uri or f"s3://{args.bucket}/{args.prefix}/code/sagemaker-source.tar.gz"
    sm = boto3.client("sagemaker", region_name=args.region)

    train_input = f"s3://{args.bucket}/{args.prefix}/input/train.csv"
    val_input = f"s3://{args.bucket}/{args.prefix}/input/validation.csv"
    output_path = f"s3://{args.bucket}/{args.prefix}/models"
    tuning_job_name = args.base_job_name

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

    print(f"HPO iniciado: {tuning_job_name}")

    if not args.wait:
        return

    terminal_states = {"Completed", "Failed", "Stopped"}
    while True:
        resp = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)
        status = resp["HyperParameterTuningJobStatus"]
        print(f"status={status}")
        if status in terminal_states:
            if status != "Completed":
                raise RuntimeError(f"HPO terminou com status {status}")
            best = resp.get("BestTrainingJob", {}).get("TrainingJobName", "")
            print(f"HPO concluído. Best training job: {best}")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
