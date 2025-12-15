"""Health check endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_model_manager, get_settings
from src.models.model_manager import ModelManager
from config.settings import Settings


router = APIRouter(
    prefix="/api/health",
    tags=["health"],
)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(default="1.0.0", description="API version")
    database: str = Field(..., description="Database status")
    models: Dict[str, Any] = Field(..., description="Model manager status")


@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and component health information.

    Returns:
        HealthResponse with status information
    """
    # Get model manager summary
    model_summary = model_manager.get_summary()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        database="connected",
        models={
            "total_tickers": model_summary["total_tickers"],
            "total_models": model_summary["total_models"],
            "trained_models": model_summary["trained_models"],
        },
    )


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check(
    model_manager: ModelManager = Depends(get_model_manager),
) -> Dict[str, Any]:
    """
    Readiness check endpoint.

    Verifies that the service is ready to handle requests.

    Returns:
        Dictionary with readiness status
    """
    model_summary = model_manager.get_summary()

    is_ready = model_summary["trained_models"] > 0

    return {
        "ready": is_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "details": {
            "trained_models": model_summary["trained_models"],
            "total_models": model_summary["total_models"],
        },
    }


@router.get("/live", response_model=Dict[str, str])
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check endpoint.

    Simple endpoint to verify the service is running.

    Returns:
        Dictionary with liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }
