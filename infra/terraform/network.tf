# VPC for the harvester task. The Fargate task runs in PRIVATE subnets; egress to
# the few external services it needs (Tavily, Discord, the Fediverse APIs) goes via a
# NAT gateway, while AWS services (Bedrock, SageMaker, DynamoDB, ECR, S3, Secrets,
# Logs) are reached over VPC endpoints so that traffic never leaves the VPC.
# This pairs with the in-code SSRF guard (assert_safe_url) to protect the instance
# metadata endpoint (SECURITY.md §2 / plan §5 "VPC egress").

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  private_subnets = [cidrsubnet(var.vpc_cidr, 4, 0), cidrsubnet(var.vpc_cidr, 4, 1)]
  public_subnets  = [cidrsubnet(var.vpc_cidr, 4, 8), cidrsubnet(var.vpc_cidr, 4, 9)]
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true # required for VPC endpoint private DNS
  tags                 = { Name = "${var.name_prefix}-vpc" }
}

# ---- Subnets --------------------------------------------------------------

resource "aws_subnet" "private" {
  count             = length(local.private_subnets)
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_subnets[count.index]
  availability_zone = local.azs[count.index]
  tags              = { Name = "${var.name_prefix}-private-${count.index}" }
}

resource "aws_subnet" "public" {
  count                   = length(local.public_subnets)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnets[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.name_prefix}-public-${count.index}" }
}

# ---- Internet + NAT -------------------------------------------------------

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.name_prefix}-igw" }
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.name_prefix}-nat-eip" }
}

# Single NAT gateway (one AZ) — fine for a low-volume hourly batch job; bump to
# one-per-AZ if HA egress is ever needed.
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${var.name_prefix}-nat" }
  depends_on    = [aws_internet_gateway.main]
}

# ---- Route tables ---------------------------------------------------------

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.name_prefix}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${var.name_prefix}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ---- Security groups ------------------------------------------------------

# The task: all egress allowed (it needs NAT egress to Tavily/Discord/Fediverse and
# HTTPS to the interface endpoints); no ingress (nothing connects to it).
resource "aws_security_group" "task" {
  name_prefix = "${var.name_prefix}-task-"
  vpc_id      = aws_vpc.main.id
  description = "Harvester Fargate task: egress only."

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name_prefix}-task-sg" }
  lifecycle { create_before_destroy = true }
}

# Interface VPC endpoints accept HTTPS from within the VPC.
resource "aws_security_group" "vpce" {
  name_prefix = "${var.name_prefix}-vpce-"
  vpc_id      = aws_vpc.main.id
  description = "Interface VPC endpoints: HTTPS from the VPC."

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = { Name = "${var.name_prefix}-vpce-sg" }
  lifecycle { create_before_destroy = true }
}

# ---- VPC endpoints --------------------------------------------------------

# Gateway endpoints (free; route-table based): S3 + DynamoDB.
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags              = { Name = "${var.name_prefix}-vpce-s3" }
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region}.dynamodb"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
  tags              = { Name = "${var.name_prefix}-vpce-dynamodb" }
}

# Interface endpoints (ENI per AZ, private DNS): the runtime APIs the task calls.
# sagemaker-runtime is only added when the ranker endpoint is deployed (and the
# task actually calls SageMaker) — otherwise it's omitted.
locals {
  interface_endpoints = merge({
    bedrock_runtime = "com.amazonaws.${var.region}.bedrock-runtime"
    ecr_api         = "com.amazonaws.${var.region}.ecr.api"
    ecr_dkr         = "com.amazonaws.${var.region}.ecr.dkr"
    secretsmanager  = "com.amazonaws.${var.region}.secretsmanager"
    logs            = "com.amazonaws.${var.region}.logs"
    }, var.enable_sagemaker_ranker ? {
    sagemaker_runtime = "com.amazonaws.${var.region}.sagemaker-runtime"
  } : {})
}

resource "aws_vpc_endpoint" "interface" {
  for_each            = local.interface_endpoints
  vpc_id              = aws_vpc.main.id
  service_name        = each.value
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.vpce.id]
  private_dns_enabled = true
  tags                = { Name = "${var.name_prefix}-vpce-${each.key}" }
}
