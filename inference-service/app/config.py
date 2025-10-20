"""
Configuration Module for Inference Service

Uses Pydantic Settings for type-safe configuration management.
Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Service configuration loaded from environment variables.
    """
    
    # Service Configuration
    SERVICE_NAME: str = Field(
        default="sentiment-inference-service",
        description="Name of the inference service"
    )
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field(
        default="mlruns",
        description="MLflow tracking URI (local path or remote server)"
    )
    
    MODEL_URI: str = Field(
        default="models:/prod-sentiment-classifier/Production",
        description=(
            "MLflow Model Registry URI for loading the production model"
        )
    )
    
    # API Configuration
    MAX_BATCH_SIZE: int = Field(
        default=100,
        description="Maximum number of texts allowed in batch prediction"
    )
    
    # Server Configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the service"
    )
    
    PORT: int = Field(
        default=8000,
        description="Port to bind the service"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Create a global settings instance
settings = Settings()

