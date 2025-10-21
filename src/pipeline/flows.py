"""
Main Prefect Flows

This module defines the primary workflows (Prefect flows) that
orchestrate the entire MLOps pipeline.

The main flow, `retraining_flow`, ties together all tasks:
1.  Data Ingestion & Validation
2.  Model Training & Evaluation
3.  Model/Data Drift Analysis
4.  Conditional Retraining & Registration

Data Processing: Uses Polars for efficient data operations, converting
to pandas only when necessary for library compatibility (e.g., Evidently AI).
"""

import polars as pl  # type: ignore
from prefect import flow, task, get_run_logger
from prefect.context import get_run_context
import mlflow

# Import configuration and utility
from src.config.settings import settings
from src.utils.logging import get_logger

# Import tasks from submodules
from src.pipeline.tasks.data import (
    load_raw_data,
    validate_data,
    preprocess_data,
    split_data,
    load_reference_data,
    simulate_current_data,
)
from src.pipeline.tasks.train import train_model
from src.pipeline.tasks.evaluate import evaluate_model
from src.pipeline.tasks.register import register_model
from src.model_monitoring.monitoring import run_drift_analysis

# Use standard logger for module-level logging
module_logger = get_logger(__name__)


@task(name="Check Drift and Performance")
def check_drift_and_performance(
    reference_df: pl.DataFrame, current_df: pl.DataFrame
) -> bool:
    """
    Runs Evidently AI analysis and determines if retraining is needed.

    Converts Polars DataFrames to pandas for Evidently AI compatibility.

    Args:
        reference_df: The reference (golden) dataset as Polars DataFrame.
        current_df: The current (production simulation) dataset as
                   Polars DataFrame.

    Returns:
        True if retraining is triggered, False otherwise.
    """
    logger = get_run_logger()
    logger.info("Checking for data drift and model performance degradation...")

    # Convert Polars DataFrames to pandas for Evidently AI
    # This is a boundary conversion - we use efficient Polars processing
    # and convert only when interfacing with pandas-only libraries
    logger.info(
        "Converting Polars DataFrames to pandas for Evidently AI..."
    )
    reference_df_pandas = reference_df.to_pandas()
    current_df_pandas = current_df.to_pandas()

    analysis_results = run_drift_analysis(
        reference_df=reference_df_pandas,
        current_df=current_df_pandas,
        settings=settings
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
    ctx = get_run_context()  # type: ignore
    try:
        flow_name = getattr(ctx.flow_run, "name", "unknown")  # type: ignore
    except (AttributeError, TypeError):
        flow_name = "unknown"
    logger.info(f"Starting flow run: {flow_name}")
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
        reference_df=reference_df, new_raw_df=raw_df, settings=settings
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
