provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "argument-quality-harvester"
      ManagedBy = "terraform"
      Component = "harvester"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
