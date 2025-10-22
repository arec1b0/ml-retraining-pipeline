"""
Pydantic Schemas for Request and Response Validation

Defines the data models for all API endpoints, ensuring type safety
and automatic validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """
    Represents a request for a single text prediction.

    Attributes:
        text: The input string to be analyzed for sentiment.
              It must be between 1 and 5000 characters and not
              consist solely of whitespace.
    """
    
    text: str = Field(
        ...,
        description="Input text for sentiment analysis",
        min_length=1,
        max_length=5000,
        examples=["This product is amazing!"]
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """
        Validates that the input text is not empty or just whitespace.

        Args:
            v: The input text string.

        Returns:
            The validated text string.

        Raises:
            ValueError: If the text is empty or contains only whitespace.
        """
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class BatchPredictionRequest(BaseModel):
    """
    Represents a request for batch predictions on a list of texts.

    Attributes:
        texts: A list of strings to be analyzed. The list must contain
               at least one string, and each string must adhere to the
               validation rules defined in `PredictionRequest`.
    """
    
    texts: List[str] = Field(
        ...,
        description="List of texts for sentiment analysis",
        min_length=1,
        examples=[["Great product!", "Terrible service", "Just okay"]]
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """
        Validates each text in the batch to ensure it meets requirements.

        Args:
            v: The list of text strings.

        Returns:
            The validated list of text strings.

        Raises:
            ValueError: If the list is empty, a text is empty/whitespace,
                        or a text exceeds the maximum length.
        """
        if not v:
            raise ValueError("Texts list cannot be empty")

        for idx, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(
                    f"Text at index {idx} is empty or only whitespace"
                )
            if len(text) > 5000:
                raise ValueError(
                    f"Text at index {idx} exceeds maximum "
                    f"length of 5000 characters"
                )

        return v


class PredictionResponse(BaseModel):
    """
    Represents the prediction result for a single text.

    Attributes:
        text: The original input text.
        sentiment: The predicted sentiment label (e.g., "positive").
        confidence: The confidence score of the prediction, between 0.0 and 1.0.
        model_version: The version of the model that made the prediction.
    """
    
    text: str = Field(..., description="Original input text")
    sentiment: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(
        ...,
        description="Prediction confidence score",
        ge=0.0,
        le=1.0
    )
    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )


class BatchPredictionResponse(BaseModel):
    """
    Represents the response for a batch prediction request.

    Attributes:
        predictions: A list of `PredictionResponse` objects, one for each
                     input text in the batch request.
    """
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each input text"
    )


class ModelInfoResponse(BaseModel):
    """
    Represents metadata about the currently loaded model.

    Provides key information about the model being served, such as its
    name in the registry, version, and the MLflow run that produced it.

    Attributes:
        model_name: The name of the model as registered in MLflow.
        version: The specific version number of the model.
        run_id: The ID of the MLflow run that generated the model.
        model_uri: The MLflow URI used to load the model.
        stage: The deployment stage of the model (e.g., "Production").
        loaded_at: The UTC timestamp (ISO format) when the model was loaded.
    """

    model_name: str = Field(
        ...,
        description="Name of the model in MLflow Registry"
    )
    version: str = Field(..., description="Model version number")
    run_id: str = Field(
        ...,
        description="MLflow run ID that created this model"
    )
    model_uri: str = Field(..., description="MLflow model URI")
    stage: str = Field(
        ...,
        description="Model stage (e.g., Production, Staging)"
    )
    loaded_at: Optional[str] = Field(
        None,
        description="Timestamp when model was loaded"
    )


class HealthResponse(BaseModel):
    """
    Represents the health status of the inference service.

    This schema is used by the `/health` endpoint to provide a status
    that can be consumed by orchestration platforms like Kubernetes.

    Attributes:
        status: The overall health status ("healthy" or "unhealthy").
        model_loaded: A boolean indicating if the ML model is successfully loaded.
        service_name: The name of the service.
        version: The current version of the service.
    """
    
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")


class ErrorResponse(BaseModel):
    """
    Represents a standardized error response.

    Used for returning consistent error messages from the API.

    Attributes:
        error: A high-level error message.
        detail: An optional field for more detailed error information.
    """
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )

