variable "region" {
  description = "AWS region. Must be one where Bedrock Nova 2 Lite is available (in the EU that means eu-west-1 + an 'eu.'-prefixed inference-profile model id)."
  type        = string
  default     = "eu-west-1"
}

variable "name_prefix" {
  description = "Prefix for all resource names."
  type        = string
  default     = "cmv-harvester"
}

# ---- Image ----------------------------------------------------------------

variable "image_tag" {
  description = "Tag of the harvester image in ECR to run (set by CI on deploy)."
  type        = string
  default     = "latest"
}

# ---- Schedule / run sizing ------------------------------------------------

variable "schedule_expression" {
  description = "EventBridge Scheduler expression for the one-shot harvester run."
  type        = string
  default     = "rate(1 hour)"
}

variable "task_cpu" {
  description = "Fargate task CPU units (CPU-only; ranker is on SageMaker)."
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "Fargate task memory (MiB)."
  type        = number
  default     = 2048
}

# ---- Bedrock --------------------------------------------------------------

variable "bedrock_model_id" {
  description = "Bedrock model id (or inference-profile id) the graph calls. In us-east-1/us-west-2 the bare id works; in the EU use the 'eu.'-prefixed cross-region inference profile."
  type        = string
  default     = "eu.amazon.nova-2-lite-v1:0"
}

# ---- SageMaker ranker -----------------------------------------------------

variable "enable_sagemaker_ranker" {
  description = "Deploy the SageMaker GPU ranker endpoint. Set false to skip it (no GPU cost) — the task then runs with RANKER_DISABLED=1 and the A/B elimination defaults to side A. The ranker model/image vars are unused when false."
  type        = bool
  default     = true
}

variable "ranker_model_s3_uri" {
  description = "S3 URI of the packaged Qwen ranker model.tar.gz (built by infra/sagemaker/package_model.sh). Unused if enable_sagemaker_ranker=false."
  type        = string
  default     = ""
}

variable "ranker_image_uri" {
  description = "ECR DLC image URI for the SageMaker ranker container (a PyTorch GPU inference DLC). Unused if enable_sagemaker_ranker=false."
  type        = string
  default     = ""
}

variable "ranker_instance_type" {
  description = "GPU instance type for the async ranker endpoint."
  type        = string
  default     = "ml.g5.2xlarge"
}

# ---- Networking -----------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for the harvester VPC."
  type        = string
  default     = "10.20.0.0/16"
}

# ---- Cost guard -----------------------------------------------------------

variable "monthly_budget_usd" {
  description = "Monthly budget (USD) for the Bedrock + SageMaker spend alarm."
  type        = number
  default     = 100
}

variable "total_budget_usd" {
  description = "Hard cap (USD) for TOTAL account spend this month — the trial cost-guard. Covers NAT/everything, not just Bedrock/SageMaker. Alerts at 50/80/100%."
  type        = number
  default     = 20
}

variable "budget_alert_email" {
  description = "Email to notify when the budget threshold is crossed."
  type        = string
}

# ---- Secrets --------------------------------------------------------------
# Secret *values* are NOT set in Terraform (never commit them). Terraform
# creates the secret containers; populate them out of band (CLI/console).

variable "create_openai_secret" {
  description = "Create an OpenAI API key secret (needed: embeddings stay on OpenAI)."
  type        = bool
  default     = true
}
