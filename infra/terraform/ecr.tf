# ECR repo for the harvester runtime image (built/pushed by CI). Scan on push,
# immutable-ish lifecycle that expires old untagged layers.

resource "aws_ecr_repository" "harvester" {
  name                 = "${var.name_prefix}/harvester"
  image_tag_mutability = "MUTABLE" # CI moves a moving tag (e.g. latest) + an immutable sha tag

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "harvester" {
  repository = aws_ecr_repository.harvester.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 14 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 14
      }
      action = { type = "expire" }
    }]
  })
}
