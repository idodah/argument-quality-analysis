output "ecr_repository_url" {
  description = "Push the harvester image here (used by CI)."
  value       = aws_ecr_repository.harvester.repository_url
}

output "ranker_bucket" {
  description = "S3 bucket for the ranker model artifact + async I/O. Upload model.tar.gz under model/."
  value       = aws_s3_bucket.ranker.id
}

output "sagemaker_endpoint_name" {
  description = "Ranker async endpoint name (SAGEMAKER_RANKER_ENDPOINT), or null if the ranker is disabled."
  value       = var.enable_sagemaker_ranker ? aws_sagemaker_endpoint.ranker[0].name : null
}

output "dynamodb_tables" {
  description = "Seen + responses table names (DDB_SEEN_TABLE / DDB_RESPONSES_TABLE)."
  value = {
    seen      = aws_dynamodb_table.seen.name
    responses = aws_dynamodb_table.responses.name
  }
}

output "task_role_arn" {
  description = "Runtime IAM role of the harvester task."
  value       = aws_iam_role.task.arn
}

output "secret_names" {
  description = "Secrets Manager containers to populate out of band."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.name }
}

output "log_group" {
  description = "CloudWatch log group for harvester runs."
  value       = aws_cloudwatch_log_group.harvester.name
}

output "schedule_name" {
  description = "EventBridge schedule driving the harvester."
  value       = aws_scheduler_schedule.harvester.name
}
