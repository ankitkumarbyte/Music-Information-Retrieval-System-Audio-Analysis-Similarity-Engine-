"""
Rate Limiting Middleware
========================
Sliding-window rate limiter using an in-memory counter.
For production, replace the in-memory store with Redis via
``aioredis`` for distributed rate limiting across replicas.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Parameters
    ----------
    max_requests : int
        Maximum allowed requests per ``window_seconds``.
    window_seconds : int
        Rolling window duration in seconds.
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # {client_ip: deque of timestamps}
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exclude health checks from rate limiting
        if request.url.path in {"/api/v1/health", "/api/v1/metrics"}:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()
        window = self._windows[client_ip]

        # Evict timestamps outside the window
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - window[0])) + 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(self.max_requests - len(window))
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
