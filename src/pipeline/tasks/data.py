"""
Data Processing Tasks

This module contains Prefect tasks related to data loading, preprocessing,
and splitting. These tasks form the initial stages of the MLOps pipeline.

Performance: Uses Polars for efficient data processing with lazy evaluation
to handle datasets larger than RAM.
"""

import polars as pl  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    train_test_split,
)
from prefect import task
from typing import Tuple

from src.config.settings import Settings
from src.utils.logging import get_logger
from src.data_validation.validation import (  # type: ignore[import-untyped]
    run_validation_checkpoint,
)

# Initialize logger for this module
logger = get_logger(__name__)


@task(name="Load Raw Data", retries=2, retry_delay_seconds=10)
def load_raw_data(settings: Settings) -> pl.DataFrame:
    """
    Loads the raw data from the specified file path using Polars.

    Uses Polars for efficient data loading with lazy evaluation support,
    allowing processing of datasets larger than available RAM.

    Args:
        settings: The application settings object.

    Returns:
        A Polars DataFrame containing the raw data.
    """
    logger.info(f"Loading raw data from: {settings.RAW_DATA_PATH}")
    try:
        # Use Polars scan_csv for lazy loading
        # (supports larger-than-RAM datasets)
        # Collect to materialize the data - for small files this is fine
        # For very large files, keep it lazy and chain operations
        df = pl.scan_csv(settings.RAW_DATA_PATH).collect()
        if df.height == 0:
            logger.warning("Raw data file is empty.")
        logger.info(f"Loaded {df.height:,} rows and {df.width} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Raw data file not found at: {settings.RAW_DATA_PATH}")
        raise
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        raise


@task(name="Validate Data Quality", retries=1)
def validate_data(
    data_path: str,
    suite_name: str = "data_quality_suite",
    checkpoint_name: str = "raw_data_checkpoint",
    data_asset_name: str = "raw_asset",
) -> bool:
    """
    Runs a Great Expectations validation checkpoint against the loaded data.
    If validation fails, this task will fail, stopping the pipeline.
    Args:
        data_path: The path to the data to validate.
        suite_name: The name of the GE Expectation Suite to use.
        checkpoint_name: The name of the GE Checkpoint to run.
        data_asset_name: The name of the GE Data Asset.
    Returns:
        True if data validation is successful.
    Raises:
        ValueError: If data validation fails.
    """
    logger.info(
        f"Running data validation suite '{suite_name}' on {data_path}"
    )
    validation_success = run_validation_checkpoint(
        checkpoint_name=checkpoint_name,
        data_asset_name=data_asset_name,
        data_path=data_path,
        suite_name=suite_name,
    )
    if not validation_success:
        # Fail the task to stop the pipeline
        raise ValueError("Data validation failed! Halting pipeline.")
    logger.info("Data validation successful.")
    return True


@task(name="Preprocess and Save Data")
def preprocess_data(
    df: pl.DataFrame, settings: Settings
) -> pl.DataFrame:
    """
    Performs preprocessing on the raw data using Polars.
    For this project, preprocessing is minimal:
    1.  Select relevant columns.
    2.  Drop rows with missing values in 'text' or 'sentiment'.
    3.  Save the processed data to the path specified in settings.
    Args:
        df: The raw Polars DataFrame.
        settings: The application settings object.
    Returns:
        A preprocessed Polars DataFrame.
    """
    logger.info("Starting data preprocessing...")
    # Select columns (in case raw data has extra columns)
    # Polars uses immutable operations by default
    processed_df = df.select(["id", "text", "sentiment"])
    # Drop missing values - Polars returns a new DataFrame
    initial_rows = processed_df.height
    processed_df = processed_df.drop_nulls(subset=["text", "sentiment"])
    rows_dropped = initial_rows - processed_df.height
    if rows_dropped > 0:
        logger.warning(
            f"Dropped {rows_dropped} rows due to missing values."
        )
    logger.info(
        f"Processed data shape: {processed_df.height:,} rows × "
        f"{processed_df.width} columns"
    )
    # Save the processed data
    try:
        # Polars write_csv is more efficient than pandas
        processed_df.write_csv(settings.PROCESSED_DATA_PATH)
        logger.info(f"Processed data saved to: {settings.PROCESSED_DATA_PATH}")
        # In a real DVC setup, we would run 'dvc add' here
        # os.system(f"dvc add {settings.PROCESSED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise
    return processed_df


@task(name="Split Data")
def split_data(
    df: pl.DataFrame, settings: Settings
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the preprocessed data into training and testing sets.
    Converts Polars DataFrame to pandas Series for sklearn compatibility,
    as sklearn's train_test_split works with pandas/numpy data structures.
    Args:
        df: The preprocessed Polars DataFrame.
        settings: The application settings object.
    Returns:
        A tuple containing (X_train, X_test, y_train, y_test) as pandas Series.
    """
    logger.info("Splitting data into training and testing sets...")
    # Extract columns as pandas Series for sklearn compatibility
    # Using to_pandas() only for the features we need to split
    X = df.select("text").to_series().to_pandas()
    y = df.select("sentiment").to_series().to_pandas()
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.MODEL_TEST_SPLIT_SIZE,
            random_state=settings.MODEL_RANDOM_STATE,
            stratify=y,  # Ensure class balance
        )
        logger.info(f"Training set size: {len(X_train):,}")
        logger.info(f"Test set size: {len(X_test):,}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise


@task(name="Load Reference Data")
def load_reference_data(path: str) -> pl.DataFrame:
    """
    Loads the reference dataset for drift comparison using Polars.
    Args:
        path: Path to the reference data file.
    Returns:
        A Polars DataFrame containing the reference data.
    Raises:
        FileNotFoundError: If reference data file is not found.
    """
    logger.info(f"Loading reference data from: {path}")
    try:
        # Use lazy loading for potential scaling
        df = pl.scan_csv(path).collect()
        logger.info(
            f"Loaded reference data: {df.height:,} rows × "
            f"{df.width} columns"
        )
        return df
    except FileNotFoundError:
        logger.error(f"Reference data not found at {path}. Halting.")
        raise
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        raise


@task(name="Simulate Current Data Generation")
def simulate_current_data(
    reference_df: pl.DataFrame, new_raw_df: pl.DataFrame, settings: Settings
) -> pl.DataFrame:
    """
    Simulates a "current" dataset for drift analysis by running the production
    model on new raw data.
    In a real system, this task would:
    1. Fetch recent prediction logs from a production database.
    2. Join them with ground truth labels (if available).
    Here, we simulate it by:
    1. Loading the *latest* production model from MLflow.
    2. Running predictions on the *new raw data*.
    3. Returning a DataFrame in the same format as the reference data.
    Args:
        reference_df: The reference (golden) dataset as Polars DataFrame.
        new_raw_df: The new raw data to generate predictions for
                    as Polars DataFrame.
        settings: The application settings object.
    Returns:
        A Polars DataFrame containing the simulated current data
        with predictions.
    """
    logger.info("Simulating current data with predictions...")
    try:
        # 1. Load the current "Production" model
        import mlflow
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
    # Convert to pandas Series for MLflow sklearn model compatibility
    X_current = new_raw_df.select("text").to_series().to_pandas()
    y_current_truth = new_raw_df.select("sentiment").to_series()
    # 3. Generate predictions
    y_current_pred = prod_model.predict(X_current)
    # 4. Assemble the "current" DataFrame using Polars
    # Extract id column efficiently
    id_series = new_raw_df.select("id").to_series()
    text_series = new_raw_df.select("text").to_series()
    current_df = pl.DataFrame({
        "id": id_series,
        "text": text_series,
        "sentiment": y_current_truth,  # Ground truth
        "prediction": y_current_pred,  # Model's prediction
    })
    logger.info(
        f"Generated 'current' data simulation: {current_df.height:,} rows × "
        f"{current_df.width} columns"
    )
    return current_df
