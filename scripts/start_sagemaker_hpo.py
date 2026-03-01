#!/usr/bin/env python3
"""Inicia HPO do RandomForest no SageMaker com Script Mode."""

from __future__ import annotations

import argparse

from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter


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
    parser.add_argument("--source-dir", default="scripts/sagemaker", help="Pasta com train_rf.py")
    args = parser.parse_args()

    session = Session(boto_region_name=args.region)

    estimator = SKLearn(
        entry_point="train_rf.py",
        source_dir=args.source_dir,
        role=args.role_arn,
        framework_version="1.2-1",
        py_version="py3",
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=f"s3://{args.bucket}/{args.prefix}/models",
        base_job_name=args.base_job_name,
        sagemaker_session=session,
        max_run=args.max_runtime_seconds,
        hyperparameters={
            "target": args.target,
            "random-state": 42,
            "n-jobs": -1,
        },
    )

    hyperparameter_ranges = {
        "n-estimators": IntegerParameter(20, 200),
        "max-depth": IntegerParameter(3, 20),
        "min-samples-split": IntegerParameter(2, 60),
        "min-samples-leaf": IntegerParameter(1, 20),
        "max-features": ContinuousParameter(0.3, 1.0),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:roc_auc",
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=[{"Name": "validation:roc_auc", "Regex": r"validation:roc_auc=([0-9\.]+)"}],
        objective_type="Maximize",
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
    )

    train_input = f"s3://{args.bucket}/{args.prefix}/input/train.csv"
    val_input = f"s3://{args.bucket}/{args.prefix}/input/validation.csv"

    tuner.fit(
        inputs={
            "train": train_input,
            "validation": val_input,
        },
        wait=True,
        logs=True,
    )

    print("HPO concluído")
    print(f"Tuning Job: {tuner.latest_tuning_job.name}")
    print(f"Best Training Job: {tuner.best_training_job()}")


if __name__ == "__main__":
    main()
