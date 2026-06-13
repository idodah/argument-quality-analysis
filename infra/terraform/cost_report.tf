# Daily spend-summary email. A Lambda reads month-to-date cost from Cost Explorer
# and the harvester schedule state, then publishes a summary to the budget SNS
# topic (which already emails the operator). EventBridge fires it once a day.
# This runs INSIDE AWS — where the creds + cost data live — unlike a cloud agent
# or local cron, neither of which can reach this account on a schedule.

variable "enable_daily_cost_report" {
  description = "Deploy the daily cost-summary Lambda + schedule."
  type        = bool
  default     = true
}

variable "cost_warn_usd" {
  description = "Month-to-date spend (USD) above which the daily report shouts a warning."
  type        = number
  default     = 15
}

# ---- Lambda source -------------------------------------------------------

locals {
  cost_report_src = <<-PY
    import os, json, datetime, boto3

    SNS_TOPIC = os.environ["SNS_TOPIC_ARN"]
    SCHEDULE  = os.environ["SCHEDULE_NAME"]
    WARN_USD  = float(os.environ.get("WARN_USD", "15"))
    BUDGET    = os.environ.get("TOTAL_BUDGET", "20")
    TRIAL_END = os.environ.get("TRIAL_END", "2026-06-18")

    def handler(event, context):
        ce  = boto3.client("ce", region_name="us-east-1")  # Cost Explorer is global/us-east-1
        sch = boto3.client("scheduler")
        sns = boto3.client("sns")

        today = datetime.date.today()
        start = today.replace(day=1).isoformat()
        end   = (today + datetime.timedelta(days=1)).isoformat()
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY", Metrics=["UnblendedCost"],
            GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
        )
        groups = resp["ResultsByTime"][0]["Groups"]
        total = sum(float(g["Metrics"]["UnblendedCost"]["Amount"]) for g in groups)
        top = sorted(groups, key=lambda g: -float(g["Metrics"]["UnblendedCost"]["Amount"]))[:5]
        breakdown = "\n".join(
            f"  {g['Keys'][0]}: $" + f"{float(g['Metrics']['UnblendedCost']['Amount']):.2f}"
            for g in top if float(g["Metrics"]["UnblendedCost"]["Amount"]) > 0
        ) or "  (no billable spend yet)"

        try:
            state = sch.get_schedule(Name=SCHEDULE)["State"]
        except Exception as e:
            state = f"unknown ({e})"

        warn = ""
        if total >= WARN_USD:
            warn = (f"\n\n⚠️  WARNING: month-to-date $${total:.2f} is at/over the "
                    f"$${WARN_USD:.0f} warn line (cap $${BUDGET}). Consider:\n"
                    f"   infra/pause.sh pause      # stop the hourly runs\n"
                    f"   terraform destroy         # full stop (removes NAT too)")

        days_left = (datetime.date.fromisoformat(TRIAL_END) - today).days
        trial = (f"\n\nTrial ends {TRIAL_END} ({days_left} day(s) left) — "
                 f"run `terraform destroy` to tear down." if days_left <= 5 else "")

        msg = (f"cmv-harvester daily cost report ({today.isoformat()})\n"
               f"Month-to-date total: $${total:.2f} (cap $${BUDGET})\n"
               f"Schedule: {state}\n\n"
               f"Top services:\n{breakdown}{warn}{trial}")

        sns.publish(TopicArn=SNS_TOPIC, Subject=f"cmv-harvester cost $${total:.2f}", Message=msg)
        return {"total": total, "state": state}
  PY
}

data "archive_file" "cost_report" {
  count       = var.enable_daily_cost_report ? 1 : 0
  type        = "zip"
  output_path = "${path.module}/.cost_report.zip"
  source {
    content  = local.cost_report_src
    filename = "index.py"
  }
}

# ---- IAM -----------------------------------------------------------------

data "aws_iam_policy_document" "cost_report_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "cost_report" {
  count              = var.enable_daily_cost_report ? 1 : 0
  name               = "${var.name_prefix}-cost-report"
  assume_role_policy = data.aws_iam_policy_document.cost_report_assume.json
}

data "aws_iam_policy_document" "cost_report" {
  statement {
    sid       = "CostExplorer"
    actions   = ["ce:GetCostAndUsage"]
    resources = ["*"] # Cost Explorer does not support resource-level scoping
  }
  statement {
    sid       = "ReadSchedule"
    actions   = ["scheduler:GetSchedule"]
    resources = [aws_scheduler_schedule.harvester.arn]
  }
  statement {
    sid       = "Publish"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.budget.arn]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:*"]
  }
}

resource "aws_iam_role_policy" "cost_report" {
  count  = var.enable_daily_cost_report ? 1 : 0
  name   = "cost-report"
  role   = aws_iam_role.cost_report[0].id
  policy = data.aws_iam_policy_document.cost_report.json
}

# ---- Lambda + daily schedule ---------------------------------------------

resource "aws_lambda_function" "cost_report" {
  count            = var.enable_daily_cost_report ? 1 : 0
  function_name    = "${var.name_prefix}-cost-report"
  role             = aws_iam_role.cost_report[0].arn
  runtime          = "python3.12"
  handler          = "index.handler"
  filename         = data.archive_file.cost_report[0].output_path
  source_code_hash = data.archive_file.cost_report[0].output_base64sha256
  timeout          = 30

  environment {
    variables = {
      SNS_TOPIC_ARN = aws_sns_topic.budget.arn
      SCHEDULE_NAME = aws_scheduler_schedule.harvester.name
      WARN_USD      = tostring(var.cost_warn_usd)
      TOTAL_BUDGET  = tostring(var.total_budget_usd)
      TRIAL_END     = "2026-06-18"
    }
  }
}

resource "aws_cloudwatch_event_rule" "cost_report_daily" {
  count               = var.enable_daily_cost_report ? 1 : 0
  name                = "${var.name_prefix}-cost-report-daily"
  schedule_expression = "cron(0 8 * * ? *)" # 08:00 UTC daily
}

resource "aws_cloudwatch_event_target" "cost_report" {
  count     = var.enable_daily_cost_report ? 1 : 0
  rule      = aws_cloudwatch_event_rule.cost_report_daily[0].name
  target_id = "lambda"
  arn       = aws_lambda_function.cost_report[0].arn
}

resource "aws_lambda_permission" "cost_report_events" {
  count         = var.enable_daily_cost_report ? 1 : 0
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.cost_report[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.cost_report_daily[0].arn
}
