"""
Model Loader Module

Handles loading ML models from MLflow Model Registry and caching them
in memory. Implements singleton pattern for efficient model management.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class for managing ML model loading and inference.

    This class loads a model from MLflow Model Registry on initialization
    and caches it in memory for fast inference. It also extracts and
    stores model metadata for monitoring purposes.
    """

    _instance: Optional['ModelManager'] = None
    _model = None
    _model_metadata: Dict[str, Any] = {}
    _loaded_at: Optional[str] = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model manager (only runs once due to singleton)."""
        if self._model is None:
            self.load_model()

    def load_model(self) -> None:
        """
        Loads the sentiment analysis model from the MLflow Model Registry.

        This method configures the MLflow tracking URI, loads the model specified
        in the settings, and extracts its metadata. It is called automatically
        on `ModelManager` initialization.

        Raises:
            RuntimeError: If the MLflow tracking URI is not set or if model
                          loading fails for any reason (e.g., model not found,
                          network error).
        """
        try:
            logger.info(
                f"Setting MLflow tracking URI to: "
                f"{settings.MLFLOW_TRACKING_URI}"
            )
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

            logger.info(f"Loading model from: {settings.MODEL_URI}")
            self._model = mlflow.pyfunc.load_model(settings.MODEL_URI)

            # Extract model metadata
            self._extract_model_metadata()

            # Record load time
            self._loaded_at = datetime.utcnow().isoformat()

            logger.info(
                f"Model loaded successfully: "
                f"{self._model_metadata.get('name')} "
                f"v{self._model_metadata.get('version')}"
            )

        except Exception as e:
            logger.error(
                f"Failed to load model from {settings.MODEL_URI}: {e}"
            )
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _extract_model_metadata(self) -> None:
        """
        Extracts metadata for the loaded model from the MLflow Model Registry.

        Connects to the MLflow server to fetch details like the model's version,
        the run ID that produced it, and its current deployment stage. This
        information is stored for logging and monitoring purposes.
        """
        try:
            client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

            # Parse model URI to extract name and stage
            # Format: models:/model-name/Stage or
            # models:/model-name/version-number
            uri_parts = settings.MODEL_URI.replace("models:/", "").split("/")
            model_name = uri_parts[0]
            stage_or_version = (
                uri_parts[1] if len(uri_parts) > 1 else "Production"
            )

            # Get model version details
            if stage_or_version.isdigit():
                # Specific version number provided
                model_version = client.get_model_version(
                    name=model_name,
                    version=stage_or_version
                )
            else:
                # Stage name provided (e.g., "Production")
                model_versions = client.get_latest_versions(
                    name=model_name,
                    stages=[stage_or_version]
                )
                if not model_versions:
                    raise ValueError(
                        f"No model found in stage: {stage_or_version}"
                    )
                model_version = model_versions[0]

            self._model_metadata = {
                "name": model_name,
                "version": model_version.version,
                "run_id": model_version.run_id,
                "stage": model_version.current_stage,
                "model_uri": settings.MODEL_URI
            }

            logger.info(f"Extracted model metadata: {self._model_metadata}")

        except Exception as e:
            logger.warning(f"Failed to extract full model metadata: {e}")
            # Fallback to minimal metadata
            self._model_metadata = {
                "name": "unknown",
                "version": "unknown",
                "run_id": "unknown",
                "stage": "unknown",
                "model_uri": settings.MODEL_URI
            }

    def predict(self, texts: list) -> list:
        """
        Generates sentiment predictions for a list of input texts.

        This is the core inference method. It takes a list of strings,
        formats them into a pandas DataFrame as expected by the MLflow model,
        and returns a list of structured prediction results. It also attempts
        to extract prediction probabilities to provide a confidence score.

        Args:
            texts: A list of strings to be analyzed.

        Returns:
            A list of dictionaries, where each dictionary contains the original
            text, the predicted sentiment, a confidence score, and the model
            version used for the prediction.

        Raises:
            RuntimeError: If the model is not loaded before this method is
                          called or if an error occurs during the prediction
                          process.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        try:
            import pandas as pd  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]

            # Prepare input DataFrame (model expects pandas DataFrame)
            input_df = pd.DataFrame({"text": texts})

            # Get predictions
            predictions = self._model.predict(input_df)

            # Get prediction probabilities for confidence scores
            # Note: This assumes the model has predict_proba method
            try:
                # Try to get the underlying sklearn model
                if hasattr(self._model, '_model_impl'):
                    sklearn_model = self._model._model_impl.python_model
                    if hasattr(sklearn_model, 'predict_proba'):
                        probas = sklearn_model.predict_proba(input_df)
                        confidences = np.max(probas, axis=1).tolist()
                    else:
                        # Fallback: use dummy confidence
                        confidences = [0.5] * len(predictions)
                else:
                    confidences = [0.5] * len(predictions)
            except Exception as e:
                logger.warning(f"Could not extract confidence scores: {e}")
                confidences = [0.5] * len(predictions)

            # Format results
            results = []
            for text, sentiment, confidence in zip(
                texts, predictions, confidences
            ):
                results.append({
                    "text": text,
                    "sentiment": str(sentiment),
                    "confidence": float(confidence),
                    "model_version": self._model_metadata.get(
                        "version", "unknown"
                    )
                })

            return results

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieves metadata about the currently loaded model.

        Returns:
            A dictionary containing key information such as the model's name,
            version, run ID, stage, and the timestamp of when it was loaded.
        """
        return {
            **self._model_metadata,
            "loaded_at": self._loaded_at
        }

    def is_loaded(self) -> bool:
        """
        Checks if the model has been successfully loaded into memory.

        This method is used by the health check endpoint to determine if the
        service is ready to accept prediction requests.

        Returns:
            True if the model is loaded, False otherwise.
        """
        return self._model is not None


# Global model manager instance
model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Retrieves the singleton instance of the ModelManager.

    This function acts as a factory and accessor for the `ModelManager`.
    It ensures that only one instance of the `ModelManager` is created and
    used throughout the application's lifecycle, preventing multiple models
    from being loaded into memory.

    Returns:
        The singleton `ModelManager` instance.
    """
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
