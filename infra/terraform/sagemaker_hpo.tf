locals {
  hpo_prefix             = var.hpo_prefix != "" ? var.hpo_prefix : "${var.project_name}-${var.environment}/hpo"
  hpo_train_s3_uri       = var.hpo_train_s3_uri != "" ? var.hpo_train_s3_uri : "s3://${aws_s3_bucket.artifacts.bucket}/${local.hpo_prefix}/input/train.csv"
  hpo_val_s3_uri         = var.hpo_validation_s3_uri != "" ? var.hpo_validation_s3_uri : "s3://${aws_s3_bucket.artifacts.bucket}/${local.hpo_prefix}/input/validation.csv"
  hpo_output_s3_uri      = var.hpo_output_s3_uri != "" ? var.hpo_output_s3_uri : "s3://${aws_s3_bucket.artifacts.bucket}/${local.hpo_prefix}/models"
  hpo_source_code_s3     = var.hpo_source_code_s3_uri != "" ? var.hpo_source_code_s3_uri : "s3://${aws_s3_bucket.artifacts.bucket}/${local.hpo_prefix}/code/sagemaker-source.tar.gz"
  hpo_job_name_effective = var.hpo_job_name != "" ? var.hpo_job_name : "${substr(replace(local.base_name, "_", "-"), 0, 20)}-hpo"
}

# O provider AWS do Terraform não possui recurso nativo para HPO do SageMaker.
# Este recurso Terraform dispara o HPO via script SDK (Terraform-first no gatilho).
resource "null_resource" "sagemaker_hpo" {
  count = var.hpo_enabled ? 1 : 0

  triggers = {
    hpo_job_name           = local.hpo_job_name_effective
    role_arn               = aws_iam_role.sagemaker_execution.arn
    bucket                 = aws_s3_bucket.artifacts.bucket
    prefix                 = local.hpo_prefix
    region                 = var.aws_region
    instance_type          = var.hpo_instance_type
    instance_count         = tostring(var.hpo_instance_count)
    max_jobs               = tostring(var.hpo_max_jobs)
    max_parallel_jobs      = tostring(var.hpo_max_parallel_jobs)
    max_runtime_seconds    = tostring(var.hpo_max_runtime_seconds)
    target                 = var.hpo_target_name
    source_code_s3_uri     = local.hpo_source_code_s3
    training_image_uri     = var.hpo_training_image_uri
    objective_metric_name  = var.hpo_objective_metric_name
    objective_metric_regex = var.hpo_objective_metric_regex
    train_s3_uri           = local.hpo_train_s3_uri
    validation_s3_uri      = local.hpo_val_s3_uri
    output_s3_uri          = local.hpo_output_s3_uri
  }

  provisioner "local-exec" {
    command = <<-EOT
      python ${path.root}/../../scripts/start_sagemaker_hpo.py \
        --role-arn "${self.triggers.role_arn}" \
        --bucket "${self.triggers.bucket}" \
        --prefix "${self.triggers.prefix}" \
        --region "${self.triggers.region}" \
        --instance-type "${self.triggers.instance_type}" \
        --instance-count "${self.triggers.instance_count}" \
        --max-jobs "${self.triggers.max_jobs}" \
        --max-parallel-jobs "${self.triggers.max_parallel_jobs}" \
        --max-runtime-seconds "${self.triggers.max_runtime_seconds}" \
        --base-job-name "${self.triggers.hpo_job_name}" \
        --target "${self.triggers.target}" \
        --source-dir "${path.root}/../../scripts/sagemaker"
    EOT
  }
}
