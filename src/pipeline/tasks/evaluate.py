"""
Model Evaluation Task

This module contains the Prefect task responsible for evaluating the
newly trained model. It calculates key performance metrics, logs them
to the corresponding MLflow run, and determines if the model meets
the minimum performance criteria to be considered for registration.
"""

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from prefect import task
from typing import Dict, Any

from src.config.settings import Settings
from src.utils.logging import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


@task(name="Evaluate Model")
def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    run_id: str,
    settings: Settings,
) -> Dict[str, Any]:
    """
    Evaluates the trained model on the test set and logs metrics to MLflow.

    This task performs the following:
    1.  Generates predictions on the test set.
    2.  Calculates accuracy, F1, precision, and recall.
    3.  Logs these metrics to the MLflow run specified by `run_id`.
    4.  Compares accuracy against the `MIN_TRAINING_ACCURACY` threshold.
    5.  Returns a dictionary with metrics and an eligibility flag.

    Args:
        pipeline: The trained scikit-learn pipeline.
        X_test: The test features (text data).
        y_test: The test target (sentiments).
        run_id: The MLflow run ID to log metrics against.
        settings: The application settings object.

    Returns:
        A dictionary containing:
        - "metrics": A sub-dictionary with calculated metrics.
        - "is_eligible": A boolean indicating if the model meets
                         the minimum accuracy threshold.
    """
    logger.info(f"Evaluating model from MLflow run: {run_id}")

    try:
        # 1. Generate predictions
        y_pred = pipeline.predict(X_test)

        # 2. Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }
        logger.info(f"Test set metrics: {metrics}")

        # 3. Log metrics to the *existing* MLflow run
        # We use mlflow.start_run() with an existing run_id to "re-open" it
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            
            # Log a tag to indicate this model has been evaluated
            mlflow.set_tag("evaluation_status", "success")

        # 4. Check eligibility for registration
        is_eligible = (
            metrics["accuracy"] >= settings.MIN_TRAINING_ACCURACY
        )
        
        if is_eligible:
            logger.info(
                f"Model accuracy ({metrics['accuracy']:.4f}) meets threshold "
                f"({settings.MIN_TRAINING_ACCURACY}). Model is eligible."
            )
        else:
            logger.warning(
                f"Model accuracy ({metrics['accuracy']:.4f}) is *below* "
                f"threshold ({settings.MIN_TRAINING_ACCURACY}). "
                "Model is NOT eligible."
            )
            # Log a tag to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("eligibility", "ineligible_low_accuracy")


        return {"metrics": metrics, "is_eligible": is_eligible}

    except Exception as e:
        logger.error(
            f"Error during model evaluation for run {run_id}: {e}"
        )
        # Tag the run as failed evaluation
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("evaluation_status", "failed")
        except Exception as mlflow_e:
            logger.error(f"Failed to tag MLflow run: {mlflow_e}")
        
        raise