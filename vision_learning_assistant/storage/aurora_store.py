"""Aurora pgvector data access for embedding persistence and search."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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

    async def ensure_schema(self) -> None:
        """Ensure required extension and table exist."""
        async with await psycopg.AsyncConnection.connect(self._database_url) as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await cur.execute(
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
                    await cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS embeddings_vector_idx
                        ON embeddings
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    )
                    await conn.commit()
                except Exception as exc:
                    LOGGER.warning(
                        "Schema bootstrap skipped (%s). Assuming Aurora schema already exists.",
                        exc,
                    )
                    await conn.rollback()

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
        vector_literal = _vector_literal(embedding)
        metadata_json = json.dumps(metadata or {})

        async with await psycopg.AsyncConnection.connect(self._database_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO embeddings (id, content, source_type, embedding, metadata)
                    VALUES (%s::uuid, %s, %s, %s::vector, %s::jsonb)
                    """,
                    (record_id, content, source_type, vector_literal, metadata_json),
                )
            await conn.commit()

        return record_id

    async def similarity_search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievedContext]:
        """Run cosine-similarity vector search and return ranked hits."""
        vector_literal = _vector_literal(query_embedding)

        async with await psycopg.AsyncConnection.connect(self._database_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        id::text,
                        content,
                        source_type,
                        metadata,
                        created_at,
                        (1 - (embedding <=> %s::vector)) AS score
                    FROM embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, vector_literal, limit),
                )
                rows = await cur.fetchall()

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


def _vector_literal(values: list[float]) -> str:
    """Convert Python float list to PostgreSQL vector literal."""
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"
