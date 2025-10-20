"""
Model Training Task

This module contains the Prefect task responsible for training the
sentiment analysis model. It integrates with MLflow to log parameters,
track the experiment, and log the final trained model artifact.
"""

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from prefect import task
from typing import Tuple, Any

from src.config.settings import Settings
from src.utils.logging import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


@task(name="Train Model with MLflow")
def train_model(
    X_train: pd.Series, y_train: pd.Series, settings: Settings
) -> Tuple[Any, str]:
    """
    Trains a model and logs the experiment to MLflow.

    This task encapsulates the model training process, including:
    1.  Setting up the MLflow experiment.
    2.  Starting an MLflow run.
    3.  Defining the scikit-learn pipeline (TF-IDF + Logistic Regression).
    4.  Logging hyperparameters.
    5.  Training the model.
    6.  Logging the trained model as an artifact.

    Args:
        X_train: The training features (text data).
        y_train: The training target (sentiments).
        settings: The application settings object.

    Returns:
        A tuple containing:
        - The trained scikit-learn pipeline object.
        - The MLflow run ID for this training run.
    """
    logger.info("Setting up MLflow experiment...")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    logger.info("Starting new MLflow run for model training...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        try:
            # 1. Define Model Parameters
            # These are hardcoded for this baseline but could be
            # passed in or optimized (e.g., Hyperopt)
            params = {
                "tfidf__ngram_range": (1, 2),
                "tfidf__max_features": 1000,
                "logreg__C": 1.0,
                "logreg__solver": "liblinear",
                "logreg__random_state": settings.MODEL_RANDOM_STATE,
            }

            # 2. Log Parameters to MLflow
            logger.info(f"Logging parameters: {params}")
            mlflow.log_params(params)
            mlflow.log_param("test_split_size", settings.MODEL_TEST_SPLIT_SIZE)
            mlflow.set_tag("model_type", "LogisticRegression")
            mlflow.set_tag("features", "TfidfVectorizer")

            # 3. Define the scikit-learn Pipeline
            pipeline = Pipeline(
                [
                    ("tfidf", TfidfVectorizer()),
                    ("logreg", LogisticRegression()),
                ]
            )
            pipeline.set_params(**params)

            # 4. Train the Model
            logger.info("Training the model...")
            pipeline.fit(X_train, y_train)
            logger.info("Model training complete.")

            # 5. Log the Model Artifact
            # We log the model here *before* evaluation.
            # Registration will happen in a separate task *after* evaluation.
            logger.info("Logging model artifact to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",  # Saved within the run's artifact dir
                # We do *not* register it here.
            )

            return pipeline, run_id

        except Exception as e:
            logger.error(f"Error during model training in run {run_id}: {e}")
            mlflow.end_run(status="FAILED")  # Explicitly fail the MLflow run
            raise