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
        Load model from MLflow Model Registry.

        Raises:
            Exception: If model loading fails.
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
        Extract metadata from the loaded MLflow model.

        This includes model name, version, run_id, and stage from the
        Model Registry.
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
        Make predictions on a list of texts.

        Args:
            texts: List of text strings to predict.

        Returns:
            List of prediction dictionaries with sentiment and confidence.

        Raises:
            RuntimeError: If model is not loaded or prediction fails.
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
        Get model metadata information.

        Returns:
            Dictionary containing model metadata.
        """
        return {
            **self._model_metadata,
            "loaded_at": self._loaded_at
        }

    def is_loaded(self) -> bool:
        """
        Check if model is loaded and ready.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self._model is not None


# Global model manager instance
model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get or create the global ModelManager instance.

    Returns:
        The ModelManager singleton instance.
    """
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
