# ECS Fargate task that runs the one-shot harvester. CPU-only (the ranker is on
# SageMaker). Two roles, deliberately separated:
#   - execution role: what ECS/Fargate needs to START the task (pull image, write
#     logs, read the secrets it injects as env).
#   - task role:      what the harvester CODE may do at runtime (Bedrock,
#     SageMaker, DynamoDB, the ranker S3 bucket). Scoped to exact ARNs, no
#     wildcards (SECURITY.md / plan §5 "least-privilege task role").

resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ---- Trust policy (shared by both roles) ----------------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# ---- Execution role -------------------------------------------------------

resource "aws_iam_role" "execution" {
  name               = "${var.name_prefix}-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# Image pull + log writes (the AWS-managed policy covers ECR pull + CloudWatch).
resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The execution role also needs to read the secrets it injects into the
# container env (the `secrets` block below resolves them at task start).
data "aws_iam_policy_document" "execution_secrets" {
  statement {
    sid       = "ReadInjectedSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [for s in aws_secretsmanager_secret.this : s.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  name   = "read-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets.json
}

# ---- Task role (runtime permissions of the harvester code) ----------------

resource "aws_iam_role" "task" {
  name               = "${var.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

data "aws_iam_policy_document" "task" {
  # Bedrock: invoke ONLY the configured model (plus its inference profile, if the
  # id is a profile that fans out to a foundation model in this region).
  statement {
    sid     = "BedrockInvokeModel"
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = [
      "arn:aws:bedrock:${var.region}::foundation-model/${var.bedrock_model_id}",
      "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:inference-profile/${var.bedrock_model_id}",
    ]
  }

  # SageMaker: invoke ONLY the ranker async endpoint + its async-IO S3 prefixes.
  # Both are omitted when the ranker endpoint is disabled (RANKER_DISABLED path).
  dynamic "statement" {
    for_each = var.enable_sagemaker_ranker ? [1] : []
    content {
      sid       = "SageMakerInvokeRanker"
      actions   = ["sagemaker:InvokeEndpointAsync"]
      resources = [aws_sagemaker_endpoint.ranker[0].arn]
    }
  }

  dynamic "statement" {
    for_each = var.enable_sagemaker_ranker ? [1] : []
    content {
      sid     = "RankerBucketIO"
      actions = ["s3:PutObject", "s3:GetObject"]
      resources = [
        "${aws_s3_bucket.ranker.arn}/ranker-requests/*",
        "${aws_s3_bucket.ranker.arn}/async/*",
      ]
    }
  }

  # DynamoDB: only the two tables, only the ops tracking.py uses.
  statement {
    sid     = "DynamoTracking"
    actions = ["dynamodb:GetItem", "dynamodb:PutItem"]
    resources = [
      aws_dynamodb_table.seen.arn,
      aws_dynamodb_table.responses.arn,
    ]
  }

  # Read the runtime secrets (Tavily / ntfy / OpenAI) directly too, so the code
  # can fetch them by name if not injected as env.
  statement {
    sid       = "ReadSecrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [for s in aws_secretsmanager_secret.this : s.arn]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "runtime"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

# ---- Task definition ------------------------------------------------------

locals {
  image = "${aws_ecr_repository.harvester.repository_url}:${var.image_tag}"

  # Non-secret runtime config passed as plain env. When the SageMaker ranker is
  # enabled, point the code at the endpoint; when disabled, set RANKER_DISABLED so
  # agents.ranker uses the no-op ranker (defaults A-vs-B to A; no GPU needed).
  ranker_env = var.enable_sagemaker_ranker ? [
    { name = "SAGEMAKER_RANKER_ENDPOINT", value = aws_sagemaker_endpoint.ranker[0].name },
    { name = "SAGEMAKER_RANKER_INPUT_BUCKET", value = aws_s3_bucket.ranker.id },
    ] : [
    { name = "RANKER_DISABLED", value = "1" },
  ]

  environment = concat([
    { name = "AWS_REGION", value = var.region },
    { name = "BEDROCK_MODEL_ID", value = var.bedrock_model_id },
    { name = "DDB_SEEN_TABLE", value = aws_dynamodb_table.seen.name },
    { name = "DDB_RESPONSES_TABLE", value = aws_dynamodb_table.responses.name },
  ], local.ranker_env)

  # Secrets injected as env from Secrets Manager (resolved by the execution role).
  # Keys here must match what the code reads; the secret VALUE is the raw key.
  container_secrets = concat(
    [
      { name = "TAVILY_API_KEY", valueFrom = aws_secretsmanager_secret.this["tavily"].arn },
      { name = "NTFY_TOPIC", valueFrom = "${aws_secretsmanager_secret.this["ntfy"].arn}:topic::" },
      { name = "NTFY_TOKEN", valueFrom = "${aws_secretsmanager_secret.this["ntfy"].arn}:token::" },
    ],
    var.create_openai_secret ? [
      { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.this["openai"].arn },
    ] : [],
  )
}

resource "aws_ecs_task_definition" "harvester" {
  family                   = "${var.name_prefix}-harvester"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name        = "harvester"
    image       = local.image
    essential   = true
    environment = local.environment
    secrets     = local.container_secrets
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.harvester.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "harvester"
      }
    }
  }])
}
