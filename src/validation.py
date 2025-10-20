"""
Data Validation Module

This module provides functions to interact with Great Expectations (GE)
for data validation. It is designed to be called from Prefect tasks.

The core function, `run_validation_checkpoint`, loads the GE context,
retrieves a predefined checkpoint, and runs it against the data
to ensure it meets all quality expectations.
"""

import great_expectations as ge
from great_expectations.checkpoint import Checkpoint
from great_expectations.core import ExpectationSuite
from great_expectations.data_context import DataContext
from great_expectations.validator.validator import Validator
from great_expectations.exceptions import DataContextError

from src.utils.logging import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def get_ge_context() -> DataContext:
    """
    Initializes and returns the Great Expectations DataContext.

    This function provides a standardized way to access the GE project
    configuration defined in `great_expectations/great_expectations.yml`.

    Returns:
        The Great Expectations DataContext object.

    Raises:
        DataContextError: If the GE context cannot be loaded.
    """
    try:
        context = ge.get_context()
        return context
    except DataContextError as e:
        logger.error(f"Failed to load Great Expectations context. Error: {e}")
        raise


def run_validation_checkpoint(
    checkpoint_name: str,
    data_asset_name: str,
    data_path: str,
    suite_name: str,
) -> bool:
    """
    Runs a Great Expectations checkpoint to validate a batch of data.

    This function dynamically configures and runs a GE checkpoint.
    If the checkpoint doesn't exist, it creates a simple one in memory
    for the validation run.

    Args:
        checkpoint_name: The name for the checkpoint (e.g., "raw_data_checkpoint").
        data_asset_name: The name of the data asset to validate (e.g., "raw_asset").
        data_path: The file path to the data to be validated.
        suite_name: The name of the Expectation Suite to validate against.

    Returns:
        True if validation succeeded, False otherwise.

    Raises:
        Exception: If any step in the validation process fails.
    """
    try:
        context = get_ge_context()
        datasource_name = "pandas_data_source"  # As defined in notebook

        # Ensure datasource exists
        try:
            datasource = context.datasources[datasource_name]
        except KeyError:
            logger.info(f"Datasource '{datasource_name}' not found. Creating...")
            datasource = context.sources.add_pandas(datasource_name)

        # Ensure data asset exists or create it
        try:
            data_asset = datasource.get_asset(data_asset_name)
        except LookupError:
            logger.info(f"Asset '{data_asset_name}' not found. Creating...")
            data_asset = datasource.add_csv_asset(
                name=data_asset_name,
                filepath_or_buffer=data_path
            )

        # Build a batch request
        batch_request = data_asset.build_batch_request()

        # Get the checkpoint
        try:
            checkpoint = context.get_checkpoint(checkpoint_name)
        except ge.exceptions.CheckpointNotFoundError:
            logger.warning(
                f"Checkpoint '{checkpoint_name}' not found. "
                "Creating a new in-memory checkpoint for this run."
            )
            # Create a simple checkpoint config in memory
            checkpoint_config = {
                "name": checkpoint_name,
                "config_version": 1.0,
                "class_name": "Checkpoint",
                "run_name_template": "%Y%m%d-%H%M%S-validation-run",
                "validations": [
                    {
                        "batch_request": batch_request,
                        "expectation_suite_name": suite_name,
                    }
                ],
            }
            # Instantiate the checkpoint
            checkpoint = Checkpoint(data_context=context, **checkpoint_config)

        logger.info(f"Running Great Expectations checkpoint: '{checkpoint_name}'")
        validation_result = checkpoint.run()

        if not validation_result.success:
            logger.error("Data validation failed!")
            logger.error(f"Validation stats: {validation_result.statistics}")
            # Optionally: Log the specific failed expectations
            for run_result in validation_result.run_results.values():
                for result in run_result["validation_result"]["results"]:
                    if not result["success"]:
                        logger.warning(
                            f"Expectation failed: {result['expectation_config']}"
                        )
            return False
        
        logger.info("Data validation successful.")
        
        # Build data docs to visualize results
        context.build_data_docs()
        logger.info(
            "Data Docs updated. View results in "
            "great_expectations/uncommitted/data_docs/local_site/index.html"
        )
        
        return True

    except Exception as e:
        logger.error(f"An error occurred during data validation: {e}")
        raise