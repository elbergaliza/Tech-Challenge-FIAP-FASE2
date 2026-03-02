# Terraform AWS (HPO RandomForest)

Infra + execução HPO via Terraform.

## O que este diretório cria
1. Infra base:
- S3 bucket de artefatos
- IAM role para GitHub Actions (OIDC)
- IAM role de execução SageMaker
- CloudWatch log group
- AWS Budget com alerta por email

2. Execução opcional de HPO:
- disparo via `null_resource` + script SDK (quando `hpo_enabled=true`)

## Pré-requisitos
1. OIDC provider do GitHub configurado na conta (`token.actions.githubusercontent.com`).
2. Terraform >= 1.6.

## Apply de infra base (sem HPO)
```bash
cd infra/terraform
terraform init
terraform apply \
  -var="project_name=tech-challenge" \
  -var="environment=dev" \
  -var="github_repository=ORG/REPO" \
  -var="notification_email=voce@exemplo.com"
```

## Apply para criar HPO (além da infra)
Use um nome único para o job a cada execução.

```bash
terraform apply \
  -var="project_name=tech-challenge" \
  -var="environment=dev" \
  -var="github_repository=ORG/REPO" \
  -var="notification_email=voce@exemplo.com" \
  -var="hpo_enabled=true" \
  -var="hpo_job_name=rfhpo-001" \
  -var="hpo_prefix=tech-challenge-hpo" \
  -var="hpo_source_code_s3_uri=s3://<bucket>/tech-challenge-hpo/code/sagemaker-source.tar.gz"
```

## Variáveis principais de HPO
- `hpo_training_image_uri`
- `hpo_instance_type`
- `hpo_max_jobs`
- `hpo_max_parallel_jobs`
- `hpo_max_runtime_seconds`

## GitHub Actions esperado
Workflow [ml-hpo.yml](/home/thallesf/Documentos/study/Tech-Challenge-FIAP-FASE2/.github/workflows/ml-hpo.yml) já:
1. exporta split -> CSV
2. sobe CSV e código no S3
3. roda Terraform apply para criar HPO

## Secrets/Variables necessários
- Secret: `AWS_GITHUB_ACTIONS_ROLE_ARN`
- Variable: `S3_BUCKET`
- Variable: `AWS_REGION`
- Opcional: `PROJECT_NAME`, `ENVIRONMENT`, `S3_PREFIX`, `HPO_TRAINING_IMAGE_URI`
