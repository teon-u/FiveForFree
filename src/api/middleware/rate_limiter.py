"""Rate limiting middleware for API protection."""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.

    For production, consider using Redis for distributed rate limiting.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute per client
            requests_per_hour: Max requests per hour per client
            burst_limit: Max burst requests in a short window (1 second)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit

        # Store request timestamps per client: {client_id: [timestamps]}
        self._requests: Dict[str, list] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier from request."""
        # Try X-Forwarded-For first (for reverse proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _cleanup_old_requests(self) -> None:
        """Remove old request timestamps to prevent memory bloat."""
        now = time.time()

        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        hour_ago = now - 3600

        # Remove entries older than 1 hour
        for client_id in list(self._requests.keys()):
            self._requests[client_id] = [
                ts for ts in self._requests[client_id]
                if ts > hour_ago
            ]
            # Remove empty entries
            if not self._requests[client_id]:
                del self._requests[client_id]

    def is_allowed(self, request: Request) -> Tuple[bool, str, Dict[str, int]]:
        """
        Check if request is allowed under rate limits.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (is_allowed, reason, rate_limit_headers)
        """
        client_id = self._get_client_id(request)
        now = time.time()

        # Periodic cleanup
        self._cleanup_old_requests()

        # Get request history for this client
        requests = self._requests[client_id]

        # Calculate counts for different windows
        one_second_ago = now - 1
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        burst_count = sum(1 for ts in requests if ts > one_second_ago)
        minute_count = sum(1 for ts in requests if ts > one_minute_ago)
        hour_count = sum(1 for ts in requests if ts > one_hour_ago)

        # Prepare rate limit headers
        headers = {
            "X-RateLimit-Limit-Minute": self.requests_per_minute,
            "X-RateLimit-Remaining-Minute": max(0, self.requests_per_minute - minute_count),
            "X-RateLimit-Limit-Hour": self.requests_per_hour,
            "X-RateLimit-Remaining-Hour": max(0, self.requests_per_hour - hour_count),
        }

        # Check burst limit
        if burst_count >= self.burst_limit:
            logger.warning(f"Rate limit (burst) exceeded for {client_id}")
            return False, "Too many requests (burst limit)", headers

        # Check per-minute limit
        if minute_count >= self.requests_per_minute:
            logger.warning(f"Rate limit (minute) exceeded for {client_id}")
            return False, "Too many requests per minute", headers

        # Check per-hour limit
        if hour_count >= self.requests_per_hour:
            logger.warning(f"Rate limit (hour) exceeded for {client_id}")
            return False, "Too many requests per hour", headers

        # Record this request
        requests.append(now)

        return True, "", headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
        exclude_paths: list = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
            burst_limit: Max burst requests (1 second)
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            burst_limit=burst_limit,
        )
        self.exclude_paths = exclude_paths or ["/docs", "/redoc", "/openapi.json", "/api/health"]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiter."""
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Check rate limit
        is_allowed, reason, headers = self.limiter.is_allowed(request)

        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": reason,
                    "retry_after": 60,  # Suggest retry after 1 minute
                },
                headers={k: str(v) for k, v in headers.items()},
            )

        # Process request and add rate limit headers to response
        response = await call_next(request)

        for key, value in headers.items():
            response.headers[key] = str(value)

        return response
