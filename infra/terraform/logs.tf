# CloudWatch log group for the harvester task. The harvester logs only post
# ids/titles, never secrets (SECURITY.md "Logs hygiene"). Retention keeps cost
# and exposure bounded.

resource "aws_cloudwatch_log_group" "harvester" {
  name              = "/ecs/${var.name_prefix}/harvester"
  retention_in_days = 30
}
