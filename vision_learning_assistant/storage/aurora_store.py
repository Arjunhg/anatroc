"""Aurora pgvector data access for embedding persistence and search."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from psycopg_pool import ConnectionPool

import psycopg

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedContext:
    """Single retrieval hit returned from Aurora similarity search."""

    id: str
    content: str
    source_type: str
    metadata: dict[str, Any]
    score: float
    created_at: datetime


class AuroraVectorStore:
    """Stores and queries vectorized context in Aurora PostgreSQL."""

    def __init__(
        self,
        database_url: str,
        enable_writes: bool = True,
        embedding_dimension: int = 1024,
    ) -> None:
        """Initialize store with connection URL and write behavior."""
        self._database_url = database_url
        self._enable_writes = enable_writes
        self._embedding_dimension = embedding_dimension

        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
        )

    async def ensure_schema(self) -> None:
        """Ensure required extension and table exist."""
        try:
            await asyncio.to_thread(self._ensure_schema_sync)
        except Exception as exc:
            LOGGER.warning("Aurora schema bootstrap skipped: %s", exc)

    async def insert_embedding(
        self,
        content: str,
        source_type: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert a content embedding record and return its ID."""
        if not self._enable_writes:
            generated_id = str(uuid.uuid4())
            LOGGER.debug("Aurora writes disabled, returning synthetic id=%s", generated_id)
            return generated_id

        if len(embedding) != self._embedding_dimension:
            raise ValueError(
                "Embedding length does not match configured Aurora vector dimension. "
                f"Expected {self._embedding_dimension}, got {len(embedding)}."
            )

        record_id = str(uuid.uuid4())
        try:
            await asyncio.to_thread(
                self._insert_embedding_sync,
                record_id,
                content,
                source_type,
                embedding,
                metadata or {},
            )
        except Exception as exc:
            LOGGER.warning("Aurora insert failed, returning synthetic id=%s (%s)", record_id, exc)
            return record_id
        return record_id

    async def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        session_id: str | None = None,
        session_run_id: str | None = None,
        source_types: list[str] | None = None,
    ) -> list[RetrievedContext]:
        """Run cosine-similarity vector search and return ranked hits."""
        try:
            rows = await asyncio.to_thread(
                self._similarity_search_sync,
                query_embedding,
                limit,
                session_id,
                session_run_id,
                source_types,
            )
        except Exception as exc:
            LOGGER.warning("Aurora similarity search failed: %s", exc)
            return []

        results: list[RetrievedContext] = []
        for row in rows:
            metadata = row[3] if isinstance(row[3], dict) else {}
            results.append(
                RetrievedContext(
                    id=row[0],
                    content=row[1],
                    source_type=row[2],
                    metadata=metadata,
                    created_at=row[4],
                    score=float(row[5]),
                )
            )
        return results

    def _ensure_schema_sync(self) -> None:
        """Create extension/table/index if they do not exist."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id UUID PRIMARY KEY,
                        content TEXT,
                        source_type TEXT,
                        embedding VECTOR({self._embedding_dimension}),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS embeddings_vector_idx
                    ON embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            conn.commit()

    def _insert_embedding_sync(
        self,
        record_id: str,
        content: str,
        source_type: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert one embedding row in a blocking context."""
        vector_literal = _vector_literal(embedding)
        metadata_json = json.dumps(metadata)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO embeddings (id, content, source_type, embedding, metadata)
                    VALUES (%s::uuid, %s, %s, %s::vector, %s::jsonb)
                    """,
                    (record_id, content, source_type, vector_literal, metadata_json),
                )
            conn.commit()

    def _similarity_search_sync(
        self,
        query_embedding: list[float],
        limit: int,
        session_id: str | None,
        session_run_id: str | None,
        source_types: list[str] | None,
    ) -> list[tuple[Any, ...]]:
        """Fetch nearest rows in a blocking context."""
        vector_literal = _vector_literal(query_embedding)
        filters: list[str] = ["created_at > NOW() - INTERVAL '10 minutes'"]
        params: list[Any] = []

        if session_id and session_id.strip():
            filters.append("metadata ->> 'session_id' = %s")
            params.append(session_id.strip())

        if session_run_id and session_run_id.strip():
            filters.append("metadata ->> 'session_run_id' = %s")
            params.append(session_run_id.strip())

        normalized_source_types = [value.strip() for value in (source_types or []) if value and value.strip()]
        if normalized_source_types:
            filters.append("source_type = ANY(%s)")
            params.append(normalized_source_types)

        where_clause = " AND ".join(filters)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id::text,
                        content,
                        source_type,
                        metadata,
                        created_at,
                        (1 - (embedding <=> %s::vector)) AS score
                    FROM embeddings
                    WHERE {where_clause}
                    ORDER by embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vector_literal, *params, vector_literal, limit],
                )
                return cur.fetchall()

    async def cleanup_embeddings(self) -> None:
        """Delete long-lived rows to keep storage bounded."""
        await asyncio.to_thread(self._cleanup_embeddings_sync)

    def _cleanup_embeddings_sync(self) -> None:
        """Delete rows older than retention period."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                        DELETE FROM embeddings
                        WHERE created_at < NOW() - INTERVAL '24 hours'
                    """
                )
            conn.commit()

    async def vacuum_analyze_embeddings(self) -> None:
        """Run VACUUM ANALYZE for healthier planner stats after churn."""
        await asyncio.to_thread(self._vacuum_analyze_embeddings_sync)

    def _vacuum_analyze_embeddings_sync(self) -> None:
        """Execute VACUUM ANALYZE in autocommit mode (required by PostgreSQL)."""
        with psycopg.connect(self._database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("VACUUM ANALYZE embeddings")


def _vector_literal(values: list[float]) -> str:
    """Convert Python float list to PostgreSQL vector literal."""
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"
