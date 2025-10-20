"""
Centralized Application Configuration

This module uses Pydantic's BaseSettings to load and validate application
settings from environment variables (typically stored in a .env file).

This provides a single, type-safe source of truth for all configuration
parameters, preventing misconfigurations and magic strings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings model. Loads values from .env files.
    """

    # -----------------------------------------------------------------
    # PROJECT CONFIGURATION
    # -----------------------------------------------------------------
    PROJECT_NAME: str = Field(
        default="AutomatedModelRetrainingPipeline",
        description=(
            "Defines the name for various services "
            "(e.g., Prefect, MLflow)"
        )
    )

    # -----------------------------------------------------------------
    # MLFLOW CONFIGURATION
    # -----------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = Field(
        default="mlruns",
        description=(
            "MLflow experiment tracking URI "
            "(local path or remote server)"
        )
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="SentimentModelRetraining",
        description="The name of the experiment to log runs under"
    )
    MODEL_REGISTRY_NAME: str = Field(
        default="prod-sentiment-classifier",
        description="The name used in the MLflow Model Registry"
    )

    # -----------------------------------------------------------------
    # DATA & REPORTING PATHS
    # -----------------------------------------------------------------
    RAW_DATA_PATH: str = Field(
        default="data/raw/feedback.csv",
        description="Relative path to the raw input data"
    )
    PROCESSED_DATA_PATH: str = Field(
        default="data/processed/sentiment.csv",
        description="Relative path to the processed training data"
    )
    REFERENCE_DATA_PATH: str = Field(
        default="data/reference/sentiment_reference.csv",
        description=(
            "Relative path to the reference dataset "
            "for drift detection"
        )
    )
    EVIDENTLY_REPORTS_PATH: str = Field(
        default="reports/evidently",
        description="Directory to save Evidently AI HTML reports"
    )

    # -----------------------------------------------------------------
    # MODEL TRAINING & VALIDATION THRESHOLDS
    # -----------------------------------------------------------------
    MIN_TRAINING_ACCURACY: float = Field(
        default=0.75,
        description="Minimum accuracy on test set to register a model"
    )
    DATA_DRIFT_F1_THRESHOLD: float = Field(
        default=0.5,
        description="F1 score threshold for the data drift classifier"
    )
    MODEL_PERFORMANCE_DEGRADATION_THRESHOLD: float = Field(
        default=0.05,
        description=(
            "Max allowed performance drop "
            "(e.g., 0.05 = 5%) vs. reference"
        )
    )

    # -----------------------------------------------------------------
    # MODEL HYPERPARAMETERS (Example)
    # -----------------------------------------------------------------
    # We can also store hyperparameters here to centralize them
    MODEL_TEST_SPLIT_SIZE: float = Field(
        default=0.2,
        description="Fraction of data to use for the test set"
    )
    MODEL_RANDOM_STATE: int = Field(
        default=42,
        description="Random state for reproducible splits and training"
    )

    # -----------------------------------------------------------------
    # CI/CD INTEGRATION (CT -> CD Pipeline Linking)
    # -----------------------------------------------------------------
    GITHUB_TOKEN: str = Field(
        default="",
        description="GitHub Personal Access Token for triggering CD workflows"
    )
    GITHUB_REPO_OWNER: str = Field(
        default="",
        description="GitHub repository owner (username or organization)"
    )
    GITHUB_REPO_NAME: str = Field(
        default="",
        description="GitHub repository name"
    )
    CD_WORKFLOW_NAME: str = Field(
        default="cd_pipeline.yml",
        description="Name of the CD workflow file to trigger"
    )
    ENABLE_CD_TRIGGER: bool = Field(
        default=False,
        description=(
            "Enable automatic CD pipeline trigger on model promotion"
        )
    )

    class Config:
        # This tells Pydantic to load variables from a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env


# Create a single, globally accessible settings instance
# Other modules can import this object directly:
# from src.config.settings import settings
settings = Settings()
