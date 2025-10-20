"""
Pydantic Schemas for Request and Response Validation

Defines the data models for all API endpoints, ensuring type safety
and automatic validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for single text prediction."""
    
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
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    
    texts: List[str] = Field(
        ...,
        description="List of texts for sentiment analysis",
        min_length=1,
        examples=[["Great product!", "Terrible service", "Just okay"]]
    )
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate each text in the batch."""
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
    """Response schema for single prediction."""
    
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
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each input text"
    )


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

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
    """Response schema for health check."""
    
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready"
    )
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")


class ErrorResponse(BaseModel):
    """Response schema for error messages."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )

