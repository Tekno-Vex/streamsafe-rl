# 1. The Provider: Tells Terraform we are building on AWS
provider "aws" {
  region = "us-east-1"
}

# 2. ECR Repository: For your Go Ingestion Service 
resource "aws_ecr_repository" "ingestion_repo" {
  name                 = "streamsafe-ingestion"
  image_tag_mutability = "MUTABLE"
  
  # force_delete allows us to easily destroy this repo later even if it has images inside
  force_delete         = true 
}

# 3. ECR Repository: For your Python Moderation API 
resource "aws_ecr_repository" "moderation_repo" {
  name                 = "streamsafe-moderation"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}

# 4. S3 Bucket: For your ML Models and Parquet Logs 
resource "aws_s3_bucket" "ml_artifacts" {
  # IMPORTANT: S3 bucket names must be globally unique across ALL of AWS. 
  # Change the name below to include your name or some random numbers!
  bucket        = "vivek-akash-streamsafe-rl-artifacts" 
  
  # force_destroy allows us to easily tear down the bucket later
  force_destroy = true
}