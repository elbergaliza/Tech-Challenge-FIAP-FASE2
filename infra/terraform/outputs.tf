output "s3_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

output "hpo_job_name" {
  value = var.hpo_enabled ? local.hpo_job_name_effective : null
}

output "hpo_trigger_id" {
  value = var.hpo_enabled ? null_resource.sagemaker_hpo[0].id : null
}
