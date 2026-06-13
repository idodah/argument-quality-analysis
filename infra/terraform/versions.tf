terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }

  # Remote state in S3 with a DynamoDB lock table. These must already exist
  # (bootstrap them once, out of band) — left as a partial config so the bucket /
  # table / region are supplied at `terraform init` via -backend-config, keeping
  # account-specific values out of source.
  backend "s3" {
    key     = "harvester/terraform.tfstate"
    encrypt = true
    # bucket         = "<your-tf-state-bucket>"        (via -backend-config)
    # dynamodb_table = "<your-tf-lock-table>"          (via -backend-config)
    # region         = "<region>"                      (via -backend-config)
  }
}
