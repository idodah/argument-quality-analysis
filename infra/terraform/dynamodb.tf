# Two tables behind harvester/tracking.py's DynamoDB backend (plan §8: two tables,
# cleanest map to the mark_seen / record split). Both keyed on a string `id` (the
# canonical post id). On-demand billing — the hourly job's traffic is tiny and
# spiky, so pay-per-request avoids provisioning idle capacity.

resource "aws_dynamodb_table" "seen" {
  name         = "${var.name_prefix}-seen"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }
}

resource "aws_dynamodb_table" "responses" {
  name         = "${var.name_prefix}-responses"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }
}
