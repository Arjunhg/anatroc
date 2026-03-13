"""Redis cache wrapper for short-lived session and retrieval state."""

from __future__ import annotations

import json
import logging
from typing import Any

from redis import asyncio as redis_async

LOGGER = logging.getLogger(__name__)


class RedisCache:
    """Simple async JSON cache backed by Redis."""

    def __init__(self, redis_url: str, enabled: bool = True) -> None:
        """Initialize cache client configuration."""
        self._redis_url = redis_url
        self._enabled = enabled
        self._client: redis_async.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis when caching is enabled."""
        if not self._enabled:
            return
        try:
            self._client = redis_async.from_url(self._redis_url, decode_responses=True)
            await self._client.ping()
        except Exception as exc:
            LOGGER.warning("Redis unavailable (%s). Continuing without cache.", exc)
            self._client = None
            self._enabled = False

    async def close(self) -> None:
        """Close Redis connection if present."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_json(self, key: str) -> dict[str, Any] | list[Any] | None:
        """Fetch and decode a JSON value from Redis."""
        if not self._enabled or self._client is None:
            return None
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("Invalid JSON in Redis for key=%s", key)
            return None
        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    async def set_json(self, key: str, value: dict[str, Any] | list[Any], ttl_seconds: int) -> None:
        """Serialize and store JSON value in Redis with TTL."""
        if not self._enabled or self._client is None:
            return
        await self._client.set(name=key, value=json.dumps(value), ex=ttl_seconds)
