# Secret CONTAINERS only. We never set secret values in Terraform (they'd land in
# state + git) — create the secrets here, then populate them out of band, e.g.:
#   aws secretsmanager put-secret-value --secret-id <name> --secret-string '...'
# The task pulls these at runtime via its task role (plan §5 "No baked secrets").

locals {
  # Tavily is always needed; OpenAI only because embeddings stay on OpenAI.
  # Email/SMTP is the notifier (the only outbound write the task makes).
  base_secrets = {
    tavily = "${var.name_prefix}/tavily-api-key"
    smtp   = "${var.name_prefix}/smtp"
  }
  openai_secret = var.create_openai_secret ? {
    openai = "${var.name_prefix}/openai-api-key"
  } : {}
  secrets = merge(local.base_secrets, local.openai_secret)
}

resource "aws_secretsmanager_secret" "this" {
  for_each = local.secrets
  name     = each.value
  tags     = { Name = each.value }
}
