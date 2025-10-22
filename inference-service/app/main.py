"""
FastAPI Inference Service

A production-ready API service for serving sentiment analysis predictions
from MLflow Model Registry. Designed for Kubernetes deployment with
health checks, monitoring, and batch prediction support.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.schemas import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
from app.model_loader import get_model_manager, ModelManager
from typing import Optional

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.
    Loads the ML model on startup.
    """
    # Startup
    global model_manager
    logger.info("Starting inference service...")
    logger.info(f"Service: {settings.SERVICE_NAME}")
    logger.info(f"Version: {__version__}")
    logger.info(f"Model URI: {settings.MODEL_URI}")

    try:
        model_manager = get_model_manager()
        logger.info("Model loaded successfully. Service is ready.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning(
            "Service starting without model. Health checks will fail."
        )

    yield

    # Shutdown
    logger.info("Shutting down inference service...")


# Initialize FastAPI application
app = FastAPI(
    title="Sentiment Analysis Inference Service",
    description=(
        "Production-ready ML inference service for sentiment analysis. "
        "Loads models from MLflow Model Registry and provides "
        "single and batch prediction endpoints."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Handles any uncaught exceptions that occur within the application.

    This middleware catches any exception that is not otherwise handled
    by a more specific exception handler. It logs the error and returns a
    standardized JSON response with a 500 Internal Server Error status code,
    preventing the exposure of sensitive stack trace information to the client.

    Args:
        request: The incoming request object (provided by FastAPI).
        exc: The exception that was raised.

    Returns:
        A `JSONResponse` with a 500 status code and a generic error message.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description=(
        "Returns the health status of the service and model readiness. "
        "Used by Kubernetes liveness/readiness probes."
    )
)
async def health_check():
    """
    Performs a health check of the service.

    This endpoint is designed to be used by orchestration systems like
    Kubernetes for liveness and readiness probes. It checks the overall
    service status and, most importantly, whether the machine learning
    model has been successfully loaded into memory.

    Returns:
        A `HealthResponse` object indicating the service's health status,
        model readiness, service name, and version.
    """
    is_healthy = model_manager is not None and model_manager.is_loaded()

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        service_name=settings.SERVICE_NAME,
        version=__version__
    )


@app.get(
    "/models/info",
    response_model=ModelInfoResponse,
    tags=["Models"],
    summary="Get model information",
    description=(
        "Returns metadata about the currently loaded model "
        "including version, run ID, and stage."
    )
)
async def get_model_info():
    """
    Retrieves metadata about the currently loaded machine learning model.

    This endpoint provides clients with detailed information about the active
    model, including its name from the MLflow Registry, its version, the
    specific run ID that produced it, and its current deployment stage.

    Returns:
        A `ModelInfoResponse` object containing the model's metadata.

    Raises:
        HTTPException: Returns a 503 Service Unavailable error if the model
                       has not been loaded yet, or a 500 Internal Server Error
                       if the metadata cannot be retrieved.
    """
    if model_manager is None or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        model_info = model_manager.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information."
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Single text prediction",
    description="Predict sentiment for a single text input.",
    status_code=status.HTTP_200_OK
)
async def predict(request: PredictionRequest):
    """
    Provides a sentiment prediction for a single input text.

    This endpoint takes a single text string, processes it through the loaded
    sentiment analysis model, and returns the predicted sentiment along with
a   confidence score and the model version used.

    Args:
        request: A `PredictionRequest` object containing the text to analyze.

    Returns:
        A `PredictionResponse` object with the sentiment prediction details.

    Raises:
        HTTPException: Returns a 503 Service Unavailable error if the model
                       is not loaded, or a 500 Internal Server Error if the
                       prediction process fails.
    """
    if model_manager is None or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service not ready."
        )

    try:
        logger.info(
            f"Received prediction request for text: '{request.text[:50]}...'"
        )

        # Make prediction (using batch method with single item)
        predictions = await asyncio.to_thread(
            model_manager.predict, [request.text]
        )
        result = predictions[0]

        logger.info(
            f"Prediction completed: sentiment={result['sentiment']}, "
            f"confidence={result['confidence']:.3f}"
        )

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed."
        )


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Batch text prediction",
    description=(
        f"Predict sentiment for multiple texts "
        f"(up to {settings.MAX_BATCH_SIZE} texts)."
    ),
    status_code=status.HTTP_200_OK
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Provides sentiment predictions for a batch of input texts.

    This endpoint is optimized for higher throughput by allowing multiple text
    strings to be processed in a single request. It imposes a configurable
    limit on the batch size to prevent server overload.

    Args:
        request: A `BatchPredictionRequest` object containing a list of texts.

    Returns:
        A `BatchPredictionResponse` object containing a list of prediction
        results for each corresponding input text.

    Raises:
        HTTPException:
            - 503 Service Unavailable: If the model is not loaded.
            - 400 Bad Request: If the number of texts in the request exceeds
              the `MAX_BATCH_SIZE` setting.
            - 500 Internal Server Error: If the prediction process fails.
    """
    if model_manager is None or not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service not ready."
        )

    # Check batch size
    if len(request.texts) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size {len(request.texts)} exceeds maximum "
                f"allowed size of {settings.MAX_BATCH_SIZE}"
            )
        )

    try:
        logger.info(
            f"Received batch prediction request for {len(request.texts)} texts"
        )

        # Make predictions
        predictions = await asyncio.to_thread(
            model_manager.predict, request.texts
        )

        # Convert to response models
        prediction_responses = [
            PredictionResponse(**pred) for pred in predictions
        ]

        logger.info(
            f"Batch prediction completed: {len(predictions)} predictions made"
        )

        return BatchPredictionResponse(predictions=prediction_responses)

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed."
        )


@app.get(
    "/",
    tags=["Root"],
    summary="Root endpoint",
    description="Returns basic information about the service."
)
async def root():
    """
    Serves as the root endpoint for the service.

    Provides basic information about the service, including its name, version,
    and links to key operational endpoints like documentation and health checks.

    Returns:
        A dictionary containing essential service information.
    """
    return {
        "service": settings.SERVICE_NAME,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/models/info"
    }
