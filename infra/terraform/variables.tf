variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Project name"
}

variable "environment" {
  type        = string
  description = "Environment (dev/hml/prod)"
  default     = "dev"
}

variable "github_repository" {
  type        = string
  description = "GitHub repo in format org/repo"
}

variable "notification_email" {
  type        = string
  description = "Email for budget alerts"
}

variable "monthly_budget_limit_usd" {
  type        = string
  default     = "10"
}

variable "create_budget" {
  type    = bool
  default = true
}

variable "hpo_enabled" {
  type        = bool
  description = "Quando true, cria um HyperParameter Tuning Job no apply"
  default     = false
}

variable "hpo_job_name" {
  type        = string
  description = "Nome do tuning job (deve ser único por execução)"
  default     = ""
}

variable "hpo_prefix" {
  type        = string
  description = "Prefixo S3 para input/output do HPO"
  default     = ""
}

variable "hpo_train_s3_uri" {
  type        = string
  description = "S3 URI do train.csv"
  default     = ""
}

variable "hpo_validation_s3_uri" {
  type        = string
  description = "S3 URI do validation.csv"
  default     = ""
}

variable "hpo_output_s3_uri" {
  type        = string
  description = "S3 URI de saída dos modelos do HPO"
  default     = ""
}

variable "hpo_source_code_s3_uri" {
  type        = string
  description = "S3 URI do tar.gz com train_rf.py para script mode"
  default     = ""
}

variable "hpo_training_image_uri" {
  type        = string
  description = "Imagem de treino scikit-learn (ECR)"
  default     = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
}

variable "hpo_target_name" {
  type        = string
  description = "Nome da coluna alvo"
  default     = "HOSPITALIZ"
}

variable "hpo_objective_metric_name" {
  type        = string
  description = "Nome da métrica objetivo"
  default     = "validation:roc_auc"
}

variable "hpo_objective_metric_regex" {
  type        = string
  description = "Regex para extrair métrica do log"
  default     = "validation:roc_auc=([0-9\\.]+)"
}

variable "hpo_instance_type" {
  type        = string
  default     = "ml.m5.xlarge"
}

variable "hpo_instance_count" {
  type        = number
  default     = 1
}

variable "hpo_volume_size_gb" {
  type        = number
  default     = 30
}

variable "hpo_max_runtime_seconds" {
  type        = number
  default     = 3600
}

variable "hpo_max_jobs" {
  type        = number
  default     = 12
}

variable "hpo_max_parallel_jobs" {
  type        = number
  default     = 2
}

variable "hpo_n_estimators_min" {
  type    = number
  default = 20
}

variable "hpo_n_estimators_max" {
  type    = number
  default = 200
}

variable "hpo_max_depth_min" {
  type    = number
  default = 3
}

variable "hpo_max_depth_max" {
  type    = number
  default = 20
}

variable "hpo_min_samples_split_min" {
  type    = number
  default = 2
}

variable "hpo_min_samples_split_max" {
  type    = number
  default = 60
}

variable "hpo_min_samples_leaf_min" {
  type    = number
  default = 1
}

variable "hpo_min_samples_leaf_max" {
  type    = number
  default = 20
}

variable "hpo_max_features_min" {
  type    = number
  default = 0.3
}

variable "hpo_max_features_max" {
  type    = number
  default = 1.0
}
