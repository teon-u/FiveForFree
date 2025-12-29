"""
Performance Optimization Module

Provides caching and performance monitoring:
- LRU cache for model instances
- TTL cache for prediction results
- Performance timing utilities
- Memory usage monitoring
"""

import time
import functools
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
import asyncio
from loguru import logger

from config.settings import settings


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support."""
    value: T
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def touch(self):
        self.access_count += 1
        self.last_accessed = datetime.now()


class TTLCache(Generic[T]):
    """
    Time-To-Live cache with automatic expiration.

    Thread-safe cache that automatically removes expired entries.
    """

    def __init__(
        self,
        ttl_seconds: int = 60,
        max_size: int = 1000,
        cleanup_interval: int = 60
    ):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

        # Background cleanup (optional)
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: Optional[threading.Thread] = None

    def get(self, key: str) -> Optional[T]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self._stats['hits'] += 1
                    return entry.value
                else:
                    # Remove expired entry
                    del self._cache[key]

            self._stats['misses'] += 1
            return None

    def set(self, key: str, value: T, ttl: int = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl)

        with self._lock:
            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at
            )

    def delete(self, key: str) -> bool:
        """Remove key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def _evict_oldest(self) -> None:
        """Evict oldest entry to make room."""
        if not self._cache:
            return

        # Find least recently accessed
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        del self._cache[oldest_key]
        self._stats['evictions'] += 1

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        removed = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0

            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate
            }


class LRUModelCache:
    """
    LRU (Least Recently Used) cache for model instances.

    Keeps frequently accessed models in memory while
    evicting least recently used ones when memory is tight.
    """

    def __init__(
        self,
        max_models: int = 100,
        max_memory_mb: int = 2048
    ):
        self.max_models = max_models
        self.max_memory_mb = max_memory_mb
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'loads': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get model from cache, moving to end (most recent)."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                return self._cache[key]

            self._stats['misses'] += 1
            return None

    def set(self, key: str, model: Any) -> None:
        """Add model to cache."""
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]

            # Check limits
            self._ensure_capacity()

            # Add to cache
            self._cache[key] = model
            self._stats['loads'] += 1

    def _ensure_capacity(self) -> None:
        """Ensure cache is within limits."""
        # Check model count
        while len(self._cache) >= self.max_models:
            self._evict_lru()

        # Check memory usage
        while self._get_memory_usage_mb() > self.max_memory_mb * 0.9:
            if not self._cache:
                break
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._stats['evictions'] += 1
            logger.debug(f"Evicted model from cache: {key}")

    def _get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0

            return {
                **self._stats,
                'size': len(self._cache),
                'max_models': self.max_models,
                'hit_rate': hit_rate,
                'memory_mb': self._get_memory_usage_mb(),
                'max_memory_mb': self.max_memory_mb
            }

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()


class PredictionCache:
    """
    Specialized cache for prediction results.

    Caches predictions by ticker with short TTL (1 minute).
    """

    def __init__(self, ttl_seconds: int = 60):
        self._cache = TTLCache[Dict](ttl_seconds=ttl_seconds, max_size=500)

    def get(self, ticker: str) -> Optional[Dict]:
        """Get cached prediction for ticker."""
        return self._cache.get(ticker)

    def set(self, ticker: str, prediction: Dict, ttl: int = None) -> None:
        """Cache prediction result."""
        self._cache.set(ticker, prediction, ttl)

    def invalidate(self, ticker: str) -> None:
        """Invalidate cached prediction for ticker."""
        self._cache.delete(ticker)

    def invalidate_all(self) -> None:
        """Invalidate all cached predictions."""
        self._cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return self._cache.get_stats()


@dataclass
class PerformanceMetrics:
    """Performance timing metrics."""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    details: Optional[Dict] = None


class PerformanceMonitor:
    """
    Monitors and records performance metrics.

    Tracks operation timings and provides statistics.
    """

    def __init__(self, max_records: int = 10000):
        self.max_records = max_records
        self._metrics: list[PerformanceMetrics] = []
        self._lock = threading.Lock()

    def record(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        details: Dict = None
    ) -> None:
        """Record a performance metric."""
        with self._lock:
            if len(self._metrics) >= self.max_records:
                # Remove oldest 10%
                self._metrics = self._metrics[self.max_records // 10:]

            self._metrics.append(PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=datetime.now(),
                success=success,
                details=details
            ))

    def get_stats(self, operation: str = None, last_n: int = 100) -> Dict:
        """Get performance statistics."""
        with self._lock:
            # Filter by operation if specified
            if operation:
                metrics = [m for m in self._metrics if m.operation == operation]
            else:
                metrics = self._metrics.copy()

            # Get last N
            metrics = metrics[-last_n:] if len(metrics) > last_n else metrics

            if not metrics:
                return {
                    'count': 0,
                    'avg_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0,
                    'p50_ms': 0,
                    'p95_ms': 0,
                    'p99_ms': 0,
                    'success_rate': 0
                }

            durations = sorted([m.duration_ms for m in metrics])
            success_count = sum(1 for m in metrics if m.success)

            return {
                'count': len(metrics),
                'avg_ms': sum(durations) / len(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'p50_ms': durations[len(durations) // 2],
                'p95_ms': durations[int(len(durations) * 0.95)],
                'p99_ms': durations[int(len(durations) * 0.99)],
                'success_rate': success_count / len(metrics)
            }


# Decorator for timing functions
def timed(monitor: PerformanceMonitor = None, operation_name: str = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                if monitor:
                    monitor.record(op_name, duration_ms, success)
                logger.debug(f"{op_name} took {duration_ms:.2f}ms")

        return wrapper
    return decorator


def timed_async(monitor: PerformanceMonitor = None, operation_name: str = None):
    """Decorator to time async function execution."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                if monitor:
                    monitor.record(op_name, duration_ms, success)
                logger.debug(f"{op_name} took {duration_ms:.2f}ms")

        return wrapper
    return decorator


# Global instances
_prediction_cache: Optional[PredictionCache] = None
_model_cache: Optional[LRUModelCache] = None
_performance_monitor: Optional[PerformanceMonitor] = None


def get_prediction_cache() -> PredictionCache:
    """Get global prediction cache instance."""
    global _prediction_cache
    if _prediction_cache is None:
        _prediction_cache = PredictionCache(ttl_seconds=60)
    return _prediction_cache


def get_model_cache() -> LRUModelCache:
    """Get global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = LRUModelCache(max_models=100, max_memory_mb=2048)
    return _model_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_system_metrics() -> Dict:
    """Get system resource metrics."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_total_mb': memory.total / (1024 * 1024),
        }
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {}


def get_all_cache_stats() -> Dict:
    """Get statistics from all caches."""
    return {
        'prediction_cache': get_prediction_cache().get_stats(),
        'model_cache': get_model_cache().get_stats(),
        'system': get_system_metrics()
    }


if __name__ == "__main__":
    # Demo
    print("Cache Demo")
    print("-" * 40)

    # Test TTL Cache
    cache = TTLCache[str](ttl_seconds=2)
    cache.set("test", "value")
    print(f"Cache get: {cache.get('test')}")

    time.sleep(3)
    print(f"After TTL expiry: {cache.get('test')}")

    print(f"\nCache stats: {cache.get_stats()}")

    # Test Performance Monitor
    monitor = PerformanceMonitor()

    @timed(monitor, "test_operation")
    def slow_function():
        time.sleep(0.1)
        return "done"

    for _ in range(10):
        slow_function()

    print(f"\nPerformance stats: {monitor.get_stats('test_operation')}")

    # System metrics
    print(f"\nSystem metrics: {get_system_metrics()}")
