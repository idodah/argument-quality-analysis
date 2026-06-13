# Budget alarm on Bedrock + SageMaker spend — the backstop against a cost
# amplification attack (SECURITY.md §3 / plan §5). The in-app --max-generations /
# --max-age-hours caps are the first line; this catches anything that slips past.

resource "aws_sns_topic" "budget" {
  name = "${var.name_prefix}-budget-alerts"
}

resource "aws_sns_topic_subscription" "budget_email" {
  topic_arn = aws_sns_topic.budget.arn
  protocol  = "email"
  endpoint  = var.budget_alert_email
}

resource "aws_budgets_budget" "bedrock_sagemaker" {
  name         = "${var.name_prefix}-bedrock-sagemaker"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Scope the budget to just the two services that can run up GPU/LLM spend.
  cost_filter {
    name = "Service"
    values = [
      "Amazon SageMaker",
      "Amazon Bedrock",
    ]
  }

  # Warn at 80% of actual, and at 100% of forecast.
  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.budget.arn]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_sns_topic_arns = [aws_sns_topic.budget.arn]
  }
}

# Total-account hard cap — the trial cost-guard. Unlike the Bedrock+SageMaker
# budget above, this covers EVERYTHING (NAT, data transfer, etc.), so a runaway
# cost in any service trips it. Email at 50% (early heads-up), 80%, and 100%.
resource "aws_budgets_budget" "total" {
  name         = "${var.name_prefix}-total"
  budget_type  = "COST"
  limit_amount = tostring(var.total_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = [50, 80, 100]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.budget_alert_email]
      subscriber_sns_topic_arns  = [aws_sns_topic.budget.arn]
    }
  }
}
