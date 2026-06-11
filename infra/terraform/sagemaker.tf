# SageMaker async endpoint serving the Qwen pairwise ranker (plan §3 / step 2).
# Async + scale-to-zero so there's no idle-GPU bill between the harvester's hourly
# runs; request/response are exchanged via S3 (see agents.ranker.SageMakerRanker).

# ---- S3: model artifact + async I/O --------------------------------------

# Private bucket holding the packaged model.tar.gz AND the async request/result
# objects. SSE, all public access blocked (SECURITY.md "Private model storage").
resource "aws_s3_bucket" "ranker" {
  bucket = "${var.name_prefix}-ranker-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "ranker" {
  bucket                  = aws_s3_bucket.ranker.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ranker" {
  bucket = aws_s3_bucket.ranker.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Expire the transient async request/result objects; the model artifact lives
# under model/ and is left untouched.
resource "aws_s3_bucket_lifecycle_configuration" "ranker" {
  bucket = aws_s3_bucket.ranker.id
  rule {
    id     = "expire-async-io"
    status = "Enabled"
    filter { prefix = "async/" }
    expiration { days = 7 }
  }
  rule {
    id     = "expire-ranker-requests"
    status = "Enabled"
    filter { prefix = "ranker-requests/" }
    expiration { days = 7 }
  }
}

# ---- SageMaker execution role --------------------------------------------

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${var.name_prefix}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

# Read the model artifact + write async results; pull the DLC image; write logs.
data "aws_iam_policy_document" "sagemaker_exec" {
  statement {
    sid     = "RankerBucket"
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.ranker.arn,
      "${aws_s3_bucket.ranker.arn}/*",
    ]
  }
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"] # the DLC repo ARN is account-specific; scope down if self-hosted
  }
  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_exec" {
  name   = "exec"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sagemaker_exec.json
}

# ---- Model + endpoint -----------------------------------------------------

resource "aws_sagemaker_model" "ranker" {
  name               = "${var.name_prefix}-ranker"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn

  primary_container {
    image          = var.ranker_image_uri
    model_data_url = var.ranker_model_s3_uri
    environment = {
      SAGEMAKER_PROGRAM          = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY = "/opt/ml/model/code"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "ranker" {
  name = "${var.name_prefix}-ranker"

  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.ranker.name
    instance_type          = var.ranker_instance_type
    initial_instance_count = 1
  }

  async_inference_config {
    output_config {
      s3_output_path  = "s3://${aws_s3_bucket.ranker.id}/async/output/"
      s3_failure_path = "s3://${aws_s3_bucket.ranker.id}/async/failure/"
    }
  }
}

resource "aws_sagemaker_endpoint" "ranker" {
  name                 = "${var.name_prefix}-ranker"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.ranker.name
}

# ---- Scale-to-zero autoscaling -------------------------------------------
# Async endpoints can scale to 0 instances when the queue is empty (no idle GPU
# cost), then scale up on a backlog. Target tracking on the approximate backlog.

resource "aws_appautoscaling_target" "ranker" {
  service_namespace  = "sagemaker"
  resource_id        = "endpoint/${aws_sagemaker_endpoint.ranker.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  min_capacity       = 0
  max_capacity       = 2
}

resource "aws_appautoscaling_policy" "ranker" {
  name               = "${var.name_prefix}-ranker-scale"
  service_namespace  = aws_appautoscaling_target.ranker.service_namespace
  resource_id        = aws_appautoscaling_target.ranker.resource_id
  scalable_dimension = aws_appautoscaling_target.ranker.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    target_value       = 5.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 60

    customized_metric_specification {
      metric_name = "ApproximateBacklogSizePerInstance"
      namespace   = "AWS/SageMaker"
      statistic   = "Average"
      dimensions {
        name  = "EndpointName"
        value = aws_sagemaker_endpoint.ranker.name
      }
    }
  }
}
