# -----------------------------------------------------------------
# Terraform Configuration for Local MLOps Stack
# -----------------------------------------------------------------
#
# This Terraform script defines the *local* infrastructure needed
# to run the MLOps pipeline, specifically a MinIO (S3-compatible)
# server running in Docker.
#
# This allows DVC and MLflow to store artifacts in an "S3" bucket
# without needing a real AWS account.
#
# Prerequisites:
# 1. Terraform installed (https://www.terraform.io/downloads)
# 2. Docker Desktop running
#
# Commands:
# 1. cd terraform
# 2. terraform init
# 3. terraform apply
#
# -----------------------------------------------------------------

# --- 1. Configure Providers ---

terraform {
  required_providers {
    # Provider for managing Docker resources (containers, images)
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.1"
    }
  }
}

# --- 2. Define Docker Resources ---

# Pull the MinIO image
resource "docker_image" "minio_image" {
  name = "minio/minio:RELEASE.2023-01-25T00-19-53Z"
  # Keep the image locally, don't re-pull if it exists
  keep_locally = true
}

# Create the MinIO container
resource "docker_container" "minio_container" {
  image = docker_image.minio_image.image_id
  name  = "minio-mlops-server"

  # Port mapping:
  # 9000: MinIO S3 API (what our code will talk to)
  # 9001: MinIO Web Console
  ports {
    internal = 9000
    external = 9000
  }
  ports {
    internal = 9001
    external = 9001
  }

  # Environment variables for MinIO configuration
  # WARNING: These are default, insecure credentials.
  # Do NOT use these in production.
  env = [
    "MINIO_ROOT_USER=minioadmin",
    "MINIO_ROOT_PASSWORD=minioadmin"
  ]

  # Command to start MinIO server and specify the data directory
  command = ["server", "/data", "--console-address", ":9001"]

  # Mount a local volume to persist data
  # This stores bucket data in a 'minio-data' folder
  # inside the 'terraform' directory.
  volumes {
    host_path      = "${abspath(path.cwd)}/minio-data"
    container_path = "/data"
  }

  # Health check to ensure the server is ready
  healthcheck {
    test = [
      "CMD",
      "curl",
      "-f",
      "http://localhost:9000/minio/health/live"
    ]
    interval    = "10s"
    timeout     = "5s"
    retries     = 5
    start_period = "10s"
  }
}

# --- 3. Output Variables ---

output "minio_s3_endpoint" {
  description = "The S3-compatible API endpoint for MinIO."
  value       = "http://127.0.0.1:9000"
}

output "minio_console_url" {
  description = "The URL for the MinIO web console."
  value       = "http://127.0.0.1:9001"
}

output "minio_root_user" {
  description = "MinIO Root User (Access Key)."
  value       = "minioadmin"
  sensitive   = true
}

output "minio_root_password" {
  description = "MinIO Root Password (Secret Key)."
  value       = "minioadmin"
  sensitive   = true
}