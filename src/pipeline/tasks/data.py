"""
Data Processing Tasks

This module contains Prefect tasks related to data loading, preprocessing,
and splitting. These tasks form the initial stages of the MLOps pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from prefect import task
from typing import Tuple, List

from src.config.settings import Settings
from src.utils.logging import get_logger
from src.data_validation.validation import run_validation_checkpoint

# Initialize logger for this module
logger = get_logger(__name__)


@task(name="Load Raw Data", retries=2, retry_delay_seconds=10)
def load_raw_data(settings: Settings) -> pd.DataFrame:
    """
    Loads the raw data from the specified file path.

    Args:
        settings: The application settings object.

    Returns:
        A pandas DataFrame containing the raw data.
    """
    logger.info(f"Loading raw data from: {settings.RAW_DATA_PATH}")
    try:
        df = pd.read_csv(settings.RAW_DATA_PATH)
        if df.empty:
            logger.warning("Raw data file is empty.")
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
    logger.info(f"Running data validation suite '{suite_name}' on {data_path}")
    
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
    df: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """
    Performs preprocessing on the raw data.

    For this project, preprocessing is minimal:
    1.  Select relevant columns.
    2.  Drop rows with missing values in 'text' or 'sentiment'.
    3.  Save the processed data to the path specified in settings.

    Args:
        df: The raw pandas DataFrame.
        settings: The application settings object.

    Returns:
        A preprocessed pandas DataFrame.
    """
    logger.info("Starting data preprocessing...")
    
    # Select columns (in case raw data has extra columns)
    processed_df = df[["id", "text", "sentiment"]].copy()

    # Drop missing values
    initial_rows = len(processed_df)
    processed_df.dropna(subset=["text", "sentiment"], inplace=True)
    rows_dropped = initial_rows - len(processed_df)
    
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows due to missing values.")

    # Save the processed data
    try:
        processed_df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Processed data saved to: {settings.PROCESSED_DATA_PATH}")
        # In a real DVC setup, we would run 'dvc add' here
        # os.system(f"dvc add {settings.PROCESSED_DATA_PATH}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise

    return processed_df


@task(name="Split Data")
def split_data(
    df: pd.DataFrame, settings: Settings
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the preprocessed data into training and testing sets.

    Args:
        df: The preprocessed pandas DataFrame.
        settings: The application settings object.

    Returns:
        A tuple containing (X_train, X_test, y_train, y_test).
    """
    logger.info("Splitting data into training and testing sets...")
    
    X = df["text"]
    y = df["sentiment"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.MODEL_TEST_SPLIT_SIZE,
            random_state=settings.MODEL_RANDOM_STATE,
            stratify=y,  # Ensure class balance
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise