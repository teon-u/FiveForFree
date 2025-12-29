"""Performance monitoring endpoints."""

from typing import Dict
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from loguru import logger

from src.utils.cache import (
    get_prediction_cache,
    get_model_cache,
    get_performance_monitor,
    get_all_cache_stats,
    get_system_metrics
)


router = APIRouter(
    prefix="/api/performance",
    tags=["performance"],
)


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    prediction_cache: Dict = Field(..., description="Prediction cache stats")
    model_cache: Dict = Field(..., description="Model cache stats")
    system: Dict = Field(..., description="System resource metrics")


class PerformanceStatsResponse(BaseModel):
    """Performance statistics response."""
    operation: str = Field(..., description="Operation name")
    count: int = Field(..., description="Number of operations")
    avg_ms: float = Field(..., description="Average duration in ms")
    min_ms: float = Field(..., description="Minimum duration in ms")
    max_ms: float = Field(..., description="Maximum duration in ms")
    p50_ms: float = Field(..., description="50th percentile (median)")
    p95_ms: float = Field(..., description="95th percentile")
    p99_ms: float = Field(..., description="99th percentile")
    success_rate: float = Field(..., description="Success rate (0-1)")


@router.get("/cache", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """
    Get cache statistics.

    Returns:
        Cache statistics including hit rates and sizes
    """
    stats = get_all_cache_stats()
    return CacheStatsResponse(**stats)


@router.get("/timings/{operation}")
async def get_operation_timings(
    operation: str,
    last_n: int = 100
) -> PerformanceStatsResponse:
    """
    Get performance timings for a specific operation.

    Args:
        operation: Operation name (e.g., 'predict', 'load_model')
        last_n: Number of recent operations to analyze

    Returns:
        Performance statistics
    """
    monitor = get_performance_monitor()
    stats = monitor.get_stats(operation, last_n)
    return PerformanceStatsResponse(operation=operation, **stats)


@router.get("/system")
async def get_system_stats() -> Dict:
    """
    Get system resource statistics.

    Returns:
        CPU, memory, and process metrics
    """
    return get_system_metrics()


@router.post("/cache/clear")
async def clear_caches(
    prediction_cache: bool = True,
    model_cache: bool = False
) -> Dict:
    """
    Clear caches.

    Args:
        prediction_cache: Clear prediction cache
        model_cache: Clear model cache (use with caution)

    Returns:
        Status message
    """
    cleared = []

    if prediction_cache:
        get_prediction_cache().invalidate_all()
        cleared.append("prediction_cache")

    if model_cache:
        get_model_cache().clear()
        cleared.append("model_cache")

    logger.info(f"Cleared caches: {cleared}")
    return {"cleared": cleared, "status": "ok"}


@router.get("/summary")
async def get_performance_summary() -> Dict:
    """
    Get comprehensive performance summary.

    Returns:
        Summary of all performance metrics
    """
    monitor = get_performance_monitor()
    cache_stats = get_all_cache_stats()

    # Get stats for common operations
    operations = ['predict', 'load_model', 'feature_compute', 'api_request']
    operation_stats = {}

    for op in operations:
        stats = monitor.get_stats(op, 100)
        if stats['count'] > 0:
            operation_stats[op] = {
                'count': stats['count'],
                'avg_ms': round(stats['avg_ms'], 2),
                'p95_ms': round(stats['p95_ms'], 2),
                'success_rate': round(stats['success_rate'], 3)
            }

    return {
        'cache': cache_stats,
        'operations': operation_stats,
        'status': 'healthy' if cache_stats['system'].get('memory_percent', 0) < 90 else 'warning'
    }
