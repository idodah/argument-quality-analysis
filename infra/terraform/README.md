# Terraform — harvester AWS infrastructure

Provisions the AWS resources for the deployed harvester + agent workflow
(see `../../docs/aws-deployment-plan.md` §6). The harvester runs as a one-shot
**Fargate** task on an **EventBridge** schedule; it calls **Bedrock** (text LLM),
a **SageMaker async** endpoint (Qwen ranker), and **DynamoDB** (dedup + records),
all over **VPC endpoints**, with secrets in **Secrets Manager** and a **Budgets**
alarm as the cost backstop.

## Files

| File | Resources |
|------|-----------|
| `network.tf` | VPC, public/private subnets, NAT, gateway + interface VPC endpoints, SGs |
| `ecr.tf` / `logs.tf` | Harvester image repo + lifecycle; CloudWatch log group |
| `dynamodb.tf` | `seen` + `responses` tables (on-demand, PITR, SSE) |
| `secrets.tf` | Secrets Manager **containers** (Tavily, ntfy, OpenAI) — values set out of band |
| `sagemaker.tf` | Ranker S3 bucket, model, async endpoint config (scale-to-zero), exec role |
| `ecs.tf` | Cluster, task definition, execution role + **least-privilege task role** |
| `scheduler.tf` | EventBridge schedule → `ecs:RunTask` + its role |
| `budget.tf` | Budgets alarm on Bedrock+SageMaker → SNS email |

## Prerequisites (not created here)

1. **Remote state** — an S3 bucket + DynamoDB lock table (bootstrap once, out of band).
2. **Ranker artifact** — build + upload `model.tar.gz` (see `../sagemaker/`):
   ```bash
   ../sagemaker/package_model.sh <checkpoint> s3://<ranker-bucket>/model/model.tar.gz
   ```
   The bucket is created by this stack, so upload after the first apply (or to a
   pre-existing bucket and pass its URI).
3. **Ranker DLC image** — a SageMaker PyTorch **GPU** inference DLC whose torch
   matches `uv.lock` (2.11.x). Pass its ECR URI as `ranker_image_uri`.

## Required variables

| Variable | Notes |
|----------|-------|
| `ranker_model_s3_uri` | S3 URI of the packaged `model.tar.gz` |
| `ranker_image_uri` | SageMaker GPU DLC image URI |
| `budget_alert_email` | Where budget alerts go |

Plus optional overrides in `variables.tf` (`region`, `bedrock_model_id`,
`schedule_expression`, `ranker_instance_type`, `monthly_budget_usd`, …).

## Usage

```bash
terraform init \
  -backend-config="bucket=<tf-state-bucket>" \
  -backend-config="dynamodb_table=<tf-lock-table>" \
  -backend-config="region=<region>"

terraform plan  -var ranker_model_s3_uri=... -var ranker_image_uri=... -var budget_alert_email=...
terraform apply -var ...
```

After the first apply, populate the secret values (never in Terraform):

```bash
aws secretsmanager put-secret-value --secret-id cmv-harvester/tavily-api-key --secret-string '...'
aws secretsmanager put-secret-value --secret-id cmv-harvester/openai-api-key --secret-string '...'
# ntfy is a JSON secret with topic + token keys (referenced as :topic:: / :token:: ):
aws secretsmanager put-secret-value --secret-id cmv-harvester/ntfy \
  --secret-string '{"topic":"cmv-...","token":"tk_..."}'
```

## CI/CD (GitHub Actions)

Three workflows in `.github/workflows/`, all auth'd via GitHub **OIDC** (no
static AWS keys):

| Workflow | Trigger | Does |
|----------|---------|------|
| `tests.yml` | every PR / push | runs the offline `pytest` suite (the gate) |
| `terraform.yml` | PR / push touching `infra/terraform/**` | `fmt`+`validate`+`plan` on PR, `apply` on merge to `main` |
| `deploy-image.yml` | push to `main` touching app code / Dockerfile | builds the harvester image, pushes to ECR (tags: commit sha + `latest`) |

### One-time bootstrap

The OIDC provider + CI role live in `oidc.tf`, but there's a chicken-and-egg:
the CI role is needed to run CI, but it's created *by* Terraform. So the **first**
apply is run locally (with admin creds); after that, CI uses the
`cmv-harvester-ci` role it created. (If the account already has the GitHub OIDC
provider, set `-var github_oidc_provider_exists=true` to avoid a duplicate.)

### Required GitHub repo variables (Settings → Secrets and variables → Actions)

Set as **variables** (not secrets — none are sensitive):

| Variable | Value |
|----------|-------|
| `AWS_ROLE_ARN` | the `ci_role_arn` Terraform output |
| `AWS_REGION` | deployment region |
| `TF_STATE_BUCKET` / `TF_LOCK_TABLE` | the remote-state bucket + lock table |
| `ECR_REPOSITORY` | `cmv-harvester/harvester` (the `ecr_repository_url` path part) |
| `RANKER_MODEL_S3_URI` / `RANKER_IMAGE_URI` | required Terraform inputs |
| `BUDGET_ALERT_EMAIL` | budget alert recipient |
| `CHROMA_S3_URI` | (optional) S3 location of the prebuilt `.chroma/` corpus, fetched into the image build |

## Validate locally (no AWS needed)

```bash
terraform fmt -check
terraform init -backend=false
terraform validate
```

`validate` checks the config against the AWS provider schema (it catches
unknown resources/attributes); a real `plan` needs credentials.

## Validate locally (no AWS needed)

```bash
terraform fmt -check
terraform init -backend=false
terraform validate
```

`validate` checks the config against the AWS provider schema (it catches
unknown resources/attributes); a real `plan` needs credentials.
