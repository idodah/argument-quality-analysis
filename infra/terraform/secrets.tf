# Secret CONTAINERS only. We never set secret values in Terraform (they'd land in
# state + git) — create the secrets here, then populate them out of band, e.g.:
#   aws secretsmanager put-secret-value --secret-id <name> --secret-string '...'
# The task pulls these at runtime via its task role (plan §5 "No baked secrets").

locals {
  # Tavily is always needed; OpenAI only because embeddings stay on OpenAI.
  # Both notifier backends get a secret container: the task picks one at runtime
  # via NOTIFY_BACKEND (or prefers telegram when its keys are populated). Leave
  # the unused one empty — an unpopulated secret just yields no credentials, and
  # harvester.notify falls through to whichever backend IS configured.
  base_secrets = {
    tavily   = "${var.name_prefix}/tavily-api-key"
    ntfy     = "${var.name_prefix}/ntfy"
    telegram = "${var.name_prefix}/telegram"
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
