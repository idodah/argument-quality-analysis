# GitHub Actions OIDC: lets CI assume an AWS role with NO static keys (plan §7).
# GitHub presents a short-lived OIDC token; AWS trusts it only for this repo, and
# only for the branches/PRs in the trust condition below.

variable "github_repo" {
  description = "owner/name of the GitHub repo allowed to assume the CI role."
  type        = string
  default     = "idodah/argument-quality-analysis"
}

variable "github_oidc_provider_exists" {
  description = "Set true if an account already has the token.actions OIDC provider (only one per account is allowed)."
  type        = bool
  default     = false
}

# The OIDC provider (one per account). Skip-creating if it already exists.
resource "aws_iam_openid_connect_provider" "github" {
  count          = var.github_oidc_provider_exists ? 0 : 1
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # GitHub's OIDC thumbprint is no longer validated by AWS (it uses the JWKS),
  # but the field is still required; this is GitHub's published value.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.github_oidc_provider_exists ? 1 : 0
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_arn = var.github_oidc_provider_exists ? data.aws_iam_openid_connect_provider.github[0].arn : aws_iam_openid_connect_provider.github[0].arn
}

# Trust policy: only this repo, only the default branch (apply) and pull requests
# (plan). Tighten the `sub` values if you want to restrict further.
data "aws_iam_policy_document" "ci_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${var.github_repo}:ref:refs/heads/main",
        "repo:${var.github_repo}:pull_request",
      ]
    }
  }
}

resource "aws_iam_role" "ci" {
  name               = "${var.name_prefix}-ci"
  assume_role_policy = data.aws_iam_policy_document.ci_assume.json
}

# CI permissions: push images to the harvester ECR repo, and run Terraform.
# Terraform needs broad-ish create/update on the resource types this stack
# manages — kept to the relevant services rather than "*:*".
data "aws_iam_policy_document" "ci" {
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
    ]
    resources = [aws_ecr_repository.harvester.arn]
  }

  # Terraform state access (S3 backend + DynamoDB lock).
  statement {
    sid       = "TfState"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = ["arn:aws:s3:::*-tfstate*", "arn:aws:s3:::*-tfstate*/*"]
  }
  statement {
    sid       = "TfLock"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
    resources = ["arn:aws:dynamodb:${var.region}:${data.aws_caller_identity.current.account_id}:table/*tf*lock*"]
  }

  # Manage the stack. Scoped to the services this configuration provisions; for a
  # tighter posture, replace with per-resource ARNs once they stabilize.
  statement {
    sid = "ManageStack"
    actions = [
      "ec2:*", "ecs:*", "ecr:Describe*", "ecr:List*",
      # CreateRepository + lifecycle so deploy-image CI can self-heal a missing
      # repo on a cold account (see the "Ensure ECR repository exists" step).
      "ecr:CreateRepository", "ecr:PutLifecyclePolicy", "ecr:TagResource",
      "iam:*Role*", "iam:*Policy*", "iam:GetRole", "iam:PassRole", "iam:TagRole",
      "iam:CreateOpenIDConnectProvider", "iam:GetOpenIDConnectProvider",
      "dynamodb:CreateTable", "dynamodb:Describe*", "dynamodb:UpdateTable", "dynamodb:TagResource", "dynamodb:ListTagsOfResource",
      "sagemaker:*", "s3:CreateBucket", "s3:Put*", "s3:Get*", "s3:List*",
      "secretsmanager:CreateSecret", "secretsmanager:Describe*", "secretsmanager:TagResource", "secretsmanager:GetResourcePolicy",
      "scheduler:*", "logs:*", "budgets:*", "sns:*",
      "application-autoscaling:*",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ci" {
  name   = "ci"
  role   = aws_iam_role.ci.id
  policy = data.aws_iam_policy_document.ci.json
}

output "ci_role_arn" {
  description = "Role ARN for GitHub Actions to assume via OIDC (AWS_ROLE_ARN secret/var)."
  value       = aws_iam_role.ci.arn
}
