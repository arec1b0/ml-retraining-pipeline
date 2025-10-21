"""
Model Registration Task

This module contains the Prefect task responsible for registering the
newly trained and validated model into the MLflow Model Registry.
It also handles the logic for promoting the model to a specific
stage (e.g., "Staging" or "Production") if it outperforms the
current production model.
"""

import mlflow
import requests  # type: ignore[import-untyped]
from mlflow.tracking import MlflowClient
from mlflow.entities import ModelVersion
from prefect import task, get_run_logger
from typing import Dict, Any, Optional

from src.config.settings import Settings


# Note: We use get_custom_logger() for general module-level logging
# and get_run_logger() from Prefect inside tasks for Prefect-aware logging.


def trigger_cd_pipeline(
    model_version: str,
    model_accuracy: float,
    settings: Settings,
) -> bool:
    """
    Triggers the CD pipeline via GitHub Actions workflow_dispatch API.

    This function implements the CT -> CD linkage by automatically
    triggering the deployment pipeline when a new model is promoted
    to Production.

    Args:
        model_version: The version number of the promoted model.
        model_accuracy: The accuracy of the promoted model.
        settings: The application settings object.

    Returns:
        True if the CD pipeline was successfully triggered, else False.
    """
    prefect_logger = get_run_logger()

    # Check if CD trigger is enabled
    if not settings.ENABLE_CD_TRIGGER:
        prefect_logger.info(
            "CD pipeline trigger is disabled. "
            "Set ENABLE_CD_TRIGGER=true to enable."
        )
        return False

    # Validate required GitHub configuration
    if not all([
        settings.GITHUB_TOKEN,
        settings.GITHUB_REPO_OWNER,
        settings.GITHUB_REPO_NAME
    ]):
        prefect_logger.warning(
            "GitHub configuration incomplete. "
            "Cannot trigger CD pipeline. "
            "Please set GITHUB_TOKEN, GITHUB_REPO_OWNER, "
            "and GITHUB_REPO_NAME."
        )
        return False

    try:
        # Construct GitHub Actions workflow_dispatch API URL
        api_url = (
            f"https://api.github.com/repos/"
            f"{settings.GITHUB_REPO_OWNER}/"
            f"{settings.GITHUB_REPO_NAME}/"
            f"actions/workflows/{settings.CD_WORKFLOW_NAME}/dispatches"
        )

        # Prepare the request payload
        payload = {
            "ref": "main",  # Branch to run the workflow on
            "inputs": {
                "model_version": str(model_version),
                "model_accuracy": f"{model_accuracy:.4f}",
                "trigger_source": "automated_ct_pipeline"
            }
        }

        # Prepare headers with authentication
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        prefect_logger.info(
            f"Triggering CD pipeline for model version "
            f"{model_version} with accuracy {model_accuracy:.4f}..."
        )

        # Make the API request
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=30
        )

        # Check response status
        if response.status_code == 204:
            prefect_logger.info(
                "Successfully triggered CD pipeline! "
                "Check GitHub Actions for deployment status."
            )
            return True
        else:
            prefect_logger.error(
                f"Failed to trigger CD pipeline. "
                f"Status code: {response.status_code}, "
                f"Response: {response.text}"
            )
            return False

    except requests.exceptions.RequestException as e:
        prefect_logger.error(
            f"Network error while triggering CD pipeline: {e}"
        )
        return False
    except Exception as e:
        prefect_logger.error(
            f"Unexpected error while triggering CD pipeline: {e}"
        )
        return False


@task(name="Register Model in MLflow")
def register_model(
    run_id: str,
    evaluation_results: Dict[str, Any],
    settings: Settings,
    promote_to_production: bool = False,
) -> Optional[ModelVersion]:
    """
    Registers a model from an MLflow run into the Model Registry.

    If `evaluation_results["is_eligible"]` is False, this task skips
    registration.

    If `promote_to_production` is True, it attempts to load the
    current production model, compares performance, and only promotes
    the new model if it is better.

    Args:
        run_id: The MLflow run ID of the trained model.
        evaluation_results: The dictionary from the `evaluate_model` task.
        settings: The application settings object.
        promote_to_production: Flag to control direct promotion.

    Returns:
        The MLflow ModelVersion object if registered, else None.
    """
    prefect_logger = get_run_logger()

    if not evaluation_results.get("is_eligible", False):
        prefect_logger.warning(
            f"Model from run {run_id} is not eligible for "
            f"registration. Skipping."
        )
        return None

    try:
        client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
        model_artifact_path = "model"  # As defined in the training task
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        model_name = settings.MODEL_REGISTRY_NAME
        new_accuracy = evaluation_results["metrics"]["accuracy"]

        prefect_logger.info(
            f"Registering model '{model_name}' from URI: {model_uri}"
        )

        # Register the new model version
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                "run_id": run_id,
                "accuracy": new_accuracy
            }
        )

        prefect_logger.info(
            f"Model registered as Version: {model_version.version}"
        )

        # Add a description to the model version
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=(
                f"Model trained in run {run_id} with "
                f"test accuracy: {new_accuracy:.4f}."
            ),
        )

        # --- Promotion Logic ---
        if promote_to_production:
            promote_model(
                client, model_version, new_accuracy, settings
            )
        else:
            # Default to transitioning to "Staging"
            prefect_logger.info(
                "Transitioning new model version to 'Staging'..."
            )
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=True,
            )

        return model_version

    except Exception as e:
        prefect_logger.error(f"Error during model registration: {e}")
        raise


def promote_model(
    client: MlflowClient,
    new_model_version: ModelVersion,
    new_accuracy: float,
    settings: Settings,
):
    """
    Handles the logic for promoting a new model to "Production".

    Compares the new model's performance against the current "Production"
    model. If it's better, it promotes the new one and archives the old one.
    """
    prefect_logger = get_run_logger()
    model_name = new_model_version.name
    new_version_num = new_model_version.version

    try:
        # 1. Get the current production model
        current_prod_models = client.get_latest_versions(
            model_name, stages=["Production"]
        )

        if not current_prod_models:
            # If no model is in production, promote this one
            prefect_logger.info(
                "No model currently in 'Production'. "
                "Promoting new model..."
            )
            client.transition_model_version_stage(
                name=model_name,
                version=new_version_num,
                stage="Production",
                archive_existing_versions=False,
            )

            # Trigger CD pipeline after first production model promotion
            prefect_logger.info(
                "🔗 Initiating CT -> CD pipeline linkage "
                "for first production model..."
            )
            cd_triggered = trigger_cd_pipeline(
                model_version=new_version_num,
                model_accuracy=new_accuracy,
                settings=settings
            )

            if cd_triggered:
                prefect_logger.info(
                    "✅ CD pipeline triggered successfully "
                    "for first production deployment."
                )
            else:
                prefect_logger.warning(
                    "⚠️ CD pipeline was not triggered. "
                    "Manual deployment may be required."
                )

            return

        # 2. Compare against the current production model
        current_prod_model = current_prod_models[0]
        current_prod_run_id = current_prod_model.run_id

        # Get the current model's accuracy from its run tags or metrics
        current_accuracy = 0.0
        try:
            # Try getting from tags first
            current_accuracy = float(
                current_prod_model.tags.get("accuracy", 0.0)
            )
            if current_accuracy == 0.0:
                # If not in tags, try metrics
                current_accuracy = client.get_run(
                    current_prod_run_id
                ).data.metrics["accuracy"]
        except Exception:
            prefect_logger.warning(
                f"Could not retrieve accuracy for production model "
                f"version {current_prod_model.version}. "
                f"Defaulting to 0.0."
            )

        # 3. Make promotion decision
        prefect_logger.info(
            f"Comparing models: New (v{new_version_num}, "
            f"Acc: {new_accuracy:.4f}) vs. "
            f"Production (v{current_prod_model.version}, "
            f"Acc: {current_accuracy:.4f})"
        )

        if new_accuracy > current_accuracy:
            prefect_logger.info(
                "New model is better. Promoting to 'Production' "
                "and archiving old version."
            )
            # Promote new model to Production
            client.transition_model_version_stage(
                name=model_name,
                version=new_version_num,
                stage="Production",
                archive_existing_versions=True,  # Archive old prod model
            )

            # Trigger CD pipeline after successful promotion
            prefect_logger.info(
                "🔗 Initiating CT -> CD pipeline linkage..."
            )
            cd_triggered = trigger_cd_pipeline(
                model_version=new_version_num,
                model_accuracy=new_accuracy,
                settings=settings
            )

            if cd_triggered:
                prefect_logger.info(
                    "✅ CD pipeline triggered successfully. "
                    "New model will be deployed automatically."
                )
            else:
                prefect_logger.warning(
                    "⚠️ CD pipeline was not triggered. "
                    "Manual deployment may be required."
                )
        else:
            prefect_logger.warning(
                "New model is not better than the current production "
                "model. Transitioning to 'Staging' instead."
            )
            # Just move to staging
            client.transition_model_version_stage(
                name=model_name,
                version=new_version_num,
                stage="Staging",
                archive_existing_versions=False,
            )

    except Exception as e:
        prefect_logger.error(f"Error during model promotion: {e}")
        # At least move it to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=new_version_num,
            stage="Staging",
            archive_existing_versions=False,
        )
