"""Context memory orchestration across Nova embeddings, Aurora, and Redis."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from vision_learning_assistant.nova.bedrock_client import NovaBedrockClient
from vision_learning_assistant.storage.aurora_store import AuroraVectorStore, RetrievedContext
from vision_learning_assistant.storage.redis_cache import RedisCache

LOGGER = logging.getLogger(__name__)


class ContextMemoryService:
    """Indexes and retrieves contextual memory for Nova Lite reasoning."""

    def __init__(
        self,
        nova_client: NovaBedrockClient,
        vector_store: AuroraVectorStore,
        redis_cache: RedisCache,
        redis_ttl_seconds: int,
        default_limit: int,
    ) -> None:
        """Initialize dependent services and retrieval settings."""
        self._nova_client = nova_client
        self._vector_store = vector_store
        self._redis_cache = redis_cache
        self._redis_ttl_seconds = redis_ttl_seconds
        self._default_limit = max(1, default_limit)

    async def start(self) -> None:
        """Initialize backing stores for context retrieval."""
        await self._vector_store.ensure_schema()
        await self._redis_cache.connect()

    async def close(self) -> None:
        """Close any external resources."""
        await self._redis_cache.close()

    async def index_content(
        self,
        content: str,
        source_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Embed and persist content into Aurora with optional metadata."""
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Cannot index empty content")

        embedding = await asyncio.to_thread(
            self._nova_client.embed_text,
            normalized_content,
            "GENERIC_INDEX",
        )
        record_id = await self._vector_store.insert_embedding(
            content=normalized_content,
            source_type=source_type,
            embedding=embedding,
            metadata=metadata,
        )
        return record_id

    async def retrieve(self, question: str, limit: int | None = None) -> list[RetrievedContext]:
        """Retrieve relevant memory for a question using embedding similarity."""
        normalized_question = question.strip()
        if not normalized_question:
            return []

        requested_limit = max(1, limit or self._default_limit)
        cache_key = self._cache_key(normalized_question, requested_limit)

        cached = await self._redis_cache.get_json(cache_key)
        if isinstance(cached, list):
            parsed = _parse_cached_context(cached)
            if parsed:
                return parsed

        query_embedding = await asyncio.to_thread(
            self._nova_client.embed_text,
            normalized_question,
            "DOCUMENT_RETRIEVAL",
        )
        results = await self._vector_store.similarity_search(query_embedding, requested_limit)

        await self._redis_cache.set_json(
            key=cache_key,
            value=[_serialize_context(item) for item in results],
            ttl_seconds=self._redis_ttl_seconds,
        )
        return results

    @staticmethod
    def _cache_key(question: str, limit: int) -> str:
        """Build deterministic Redis key for retrieval cache entries."""
        normalized_text = ContextMemoryService._normalize_question(question)
        digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        return f"context:retrieval:{digest}:{limit}"
    
    @staticmethod
    def _normalize_question(question: str) -> str:
        question = question.lower().strip()
        question = re.sub(r"[^\w\s]", "", question)   # remove punctuation
        question = re.sub(r"\s+", " ", question)      # collapse spaces
        return question



def _serialize_context(item: RetrievedContext) -> dict[str, Any]:
    """Convert RetrievedContext to JSON-compatible dictionary."""
    return {
        "id": item.id,
        "content": item.content,
        "source_type": item.source_type,
        "metadata": item.metadata,
        "score": item.score,
        "created_at": item.created_at.isoformat(),
    }


def _parse_cached_context(entries: list[Any]) -> list[RetrievedContext]:
    """Parse cached retrieval entries into strongly-typed context objects."""
    parsed: list[RetrievedContext] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            parsed.append(
                RetrievedContext(
                    id=str(entry.get("id", "")),
                    content=str(entry.get("content", "")),
                    source_type=str(entry.get("source_type", "")),
                    metadata=entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {},
                    score=float(entry.get("score", 0.0)),
                    created_at=_parse_datetime(str(entry.get("created_at", ""))),
                )
            )
        except Exception:
            LOGGER.debug("Skipping malformed cached context entry")
    return parsed


def _parse_datetime(value: str):
    """Parse ISO timestamp with UTC fallback."""
    from datetime import datetime, timezone

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
