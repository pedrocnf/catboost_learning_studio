variable "project_id" { type = string }
variable "region" { type = string  default = "us-central1" }
variable "service_name" { type = string default = "catboost-learning-studio" }
variable "artifact_registry_repository" { type = string default = "apps" }
variable "container_image" {
  type        = string
  description = "Full image URL, for example us-central1-docker.pkg.dev/PROJECT/apps/catboost-learning-studio:latest"
}
