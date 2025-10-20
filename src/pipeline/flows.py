"""
Main Prefect Flows

This module defines the primary workflows (Prefect flows) that
orchestrate the entire MLOps pipeline.

The main flow, `retraining_flow`, ties together all tasks:
1.  Data Ingestion & Validation
2.  Model Training & Evaluation
3.  Model/Data Drift Analysis
4.  Conditional Retraining & Registration
"""

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.context import get_run_context
from mlflow.tracking import MlflowClient

# Import configuration and utility
from src.config.settings import settings
from src.utils.logging import get_logger

# Import tasks from submodules
from src.pipeline.tasks.data import (
    load_raw_data,
    validate_data,
    preprocess_data,
    split_data,
)
from src.pipeline.tasks.train import train_model
from src.pipeline.tasks.evaluate import evaluate_model
from src.pipeline.tasks.register import register_model
from src.model_monitoring.monitoring import run_drift_analysis

# Use standard logger for module-level logging
module_logger = get_logger(__name__)


@task(name="Load Reference Data")
def load_reference_data(path: str) -> pd.DataFrame:
    """
    Loads the reference dataset for drift comparison.
    """
    logger = get_run_logger()
    logger.info(f"Loading reference data from: {path}")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Reference data not found at {path}. Halting.")
        raise


@task(name="Simulate Current Data Generation")
def simulate_current_data(
    reference_df: pd.DataFrame, new_raw_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simulates a "current" dataset for drift analysis.

    In a real system, this task would:
    1.  Fetch recent prediction logs from a production database.
    2.  Join them with ground truth labels (if available).

    Here, we simulate it by:
    1.  Loading the *latest* production model from MLflow.
    2.  Running predictions on the *new raw data*.
    3.  Returning a DataFrame in the same format as the reference data.
    """
    logger = get_run_logger()
    logger.info("Simulating current data with predictions...")

    try:
        # 1. Load the current "Production" model
        model_uri = f"models:/{settings.MODEL_REGISTRY_NAME}/Production"
        prod_model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded production model from {model_uri}")
    except Exception as e:
        logger.warning(
            f"Could not load 'Production' model (Error: {e}). "
            "This is expected on the *first ever* run. "
            "Using reference data as current data."
        )
        # On the first run, no "Production" model exists.
        # We return the reference data to prevent the flow from failing.
        # Drift analysis will show 0 drift.
        return reference_df

    # 2. Use new raw data as the "current" features
    # Note: We use the *full* raw_df, not just the processed one,
    # to simulate a real production scenario where we predict on all new data.
    X_current = new_raw_df["text"]
    y_current_truth = new_raw_df["sentiment"]

    # 3. Generate predictions
    y_current_pred = prod_model.predict(X_current)

    # 4. Assemble the "current" DataFrame
    current_df = pd.DataFrame(
        {
            "id": new_raw_df["id"],
            "text": X_current,
            "sentiment": y_current_truth,  # Ground truth
            "prediction": y_current_pred,  # Model's prediction
        }
    )
    
    logger.info(f"Generated 'current' data simulation: {current_df.shape}")
    return current_df


@task(name="Check Drift and Performance")
def check_drift_and_performance(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> bool:
    """
    Runs Evidently AI analysis and determines if retraining is needed.

    Args:
        reference_df: The reference (golden) dataset.
        current_df: The current (production simulation) dataset.

    Returns:
        True if retraining is triggered, False otherwise.
    """
    logger = get_run_logger()
    logger.info("Checking for data drift and model performance degradation...")

    analysis_results = run_drift_analysis(
        reference_df=reference_df, current_df=current_df, settings=settings
    )

    # Decision logic for retraining
    if analysis_results["data_drift_detected"]:
        logger.warning("Data drift DETECTED. Triggering retraining.")
        return True
    
    if analysis_results["model_performance_degraded"]:
        logger.warning(
            "Model performance degradation DETECTED. Triggering retraining."
        )
        return True

    logger.info("No significant drift or degradation detected.")
    return False


@flow(name="Automated Retraining Flow", log_prints=True)
def retraining_flow(force_retrain: bool = False):
    """
    The main MLOps flow for automated model retraining.

    This flow performs the following steps:
    1.  Loads new raw data (e.g., from `data/raw/feedback.csv`).
    2.  Validates the quality of the new data using Great Expectations.
    3.  Loads the reference (golden) dataset.
    4.  Simulates a "current" production dataset by running the
        production model on the new raw data.
    5.  Analyzes for data drift or model degradation using Evidently AI.
    6.  If drift/degradation is detected (or `force_retrain` is True),
        it triggers a full retraining pipeline:
        a. Preprocesses the new raw data.
        b. Splits the data into train/test sets.
        c. Trains a new model, logging to MLflow.
        d. Evaluates the new model.
        e. Registers the new model in the MLflow Registry and
           promotes it to "Production" if it's better than the old one.
    """
    logger = get_run_logger()
    ctx = get_run_context()
    logger.info(f"Starting flow run: {ctx.flow_run.name}")
    logger.info(f"Force Retrain: {force_retrain}")
    
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # --- 1. Load & Validate New Data ---
    raw_df = load_raw_data(settings=settings)
    _ = validate_data(
        data_path=settings.RAW_DATA_PATH, wait_for=[raw_df]
    )

    # --- 2. Load Reference & Simulate Current Data ---
    reference_df = load_reference_data(path=settings.REFERENCE_DATA_PATH)
    current_df = simulate_current_data(
        reference_df=reference_df, new_raw_df=raw_df
    )

    # --- 3. Check for Drift ---
    if not force_retrain:
        retrain_needed = check_drift_and_performance(
            reference_df=reference_df, current_df=current_df
        )
    else:
        logger.info("`force_retrain` is True. Skipping drift check.")
        retrain_needed = True

    # --- 4. Conditional Retraining Pipeline ---
    if not retrain_needed:
        logger.info("No retraining needed. Flow finished successfully.")
        return

    logger.info("=== RETRAINING PIPELINE INITIATED ===")

    # a. Preprocess
    processed_df = preprocess_data(df=raw_df, settings=settings)

    # b. Split
    X_train, X_test, y_train, y_test = split_data(
        df=processed_df, settings=settings
    )

    # c. Train
    pipeline, run_id = train_model(
        X_train=X_train, y_train=y_train, settings=settings
    )

    # d. Evaluate
    evaluation_results = evaluate_model(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        run_id=run_id,
        settings=settings,
    )

    # e. Register & Promote
    register_model(
        run_id=run_id,
        evaluation_results=evaluation_results,
        settings=settings,
        promote_to_production=True,  # Attempt to promote
    )

    logger.info("=== RETRAINING PIPELINE COMPLETE ===")


if __name__ == "__main__":
    """
    This script allows the flow to be run directly for development
    and testing, and also serves as the entrypoint for Prefect
    deployments.
    """
    module_logger.info("Starting flow execution from __main__...")
    
    # Run the flow with forcing
    # On the very first run, a "Production" model won't exist,
    # so `simulate_current_data` will pass, and drift check
    # will show 0 drift. We must force the first run.
    retraining_flow(force_retrain=True)