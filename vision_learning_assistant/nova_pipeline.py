"""Nova-first orchestration pipeline for screen analysis and retrieval."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vision_learning_assistant.config import AssistantConfig
from vision_learning_assistant.nova.bedrock_client import NovaBedrockClient
from vision_learning_assistant.services.context_memory import ContextMemoryService
from vision_learning_assistant.services.screen_analysis import ScreenAnalysisResult, ScreenAnalysisService
from vision_learning_assistant.storage.aurora_store import AuroraVectorStore
from vision_learning_assistant.storage.redis_cache import RedisCache

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionFrameSnapshot:
    """Most recent sampled screen frame state for a session."""

    frame_bytes: bytes
    ocr_text: str
    source_type: str
    ingested_at: datetime


@dataclass(slots=True)
class FrameIngestionResult:
    """Result of a screen frame ingestion request."""

    session_id: str
    processed: bool
    skipped_reason: str | None
    ocr_text: str
    ocr_record_id: str | None
    analysis_result: ScreenAnalysisResult | None
    ingested_at: datetime


class VisionLearningPipeline:
    """Primary Nova-backed pipeline used by CLI and backend services."""

    def __init__(self, config: AssistantConfig) -> None:
        """Initialize all service dependencies from runtime config."""
        self._config = config
        self._nova_client = NovaBedrockClient(
            region=config.aws_region,
            lite_model_id=config.nova_lite_model_id,
            embed_model_id=config.nova_embed_model_id,
            embedding_dimension=config.nova_embedding_dimension,
            text_truncation_mode=config.nova_text_truncation_mode,
        )
        self._vector_store = AuroraVectorStore(
            database_url=config.database_url,
            enable_writes=config.enable_aurora_writes,
            embedding_dimension=config.nova_embedding_dimension,
        )
        self._redis_cache = RedisCache(
            redis_url=config.redis_url,
            enabled=config.enable_redis_cache,
        )
        self._memory = ContextMemoryService(
            nova_client=self._nova_client,
            vector_store=self._vector_store,
            redis_cache=self._redis_cache,
            redis_ttl_seconds=config.redis_ttl_seconds,
            default_limit=config.max_retrieval_results,
        )
        self._screen_analysis = ScreenAnalysisService(
            nova_client=self._nova_client,
            memory_service=self._memory,
            capture_interval_seconds=config.screen_capture_interval_seconds,
        )
        self._session_frames: dict[str, SessionFrameSnapshot] = {}

    async def start(self) -> None:
        """Connect to Aurora and Redis resources."""
        LOGGER.info("Initializing context memory services")
        await self._memory.start()

    async def close(self) -> None:
        """Close external resources."""
        LOGGER.info("Closing context memory services")
        await self._memory.close()

    async def analyze_screen_file(
        self,
        session_id: str,
        frame_path: str,
        prompt: str,
    ) -> ScreenAnalysisResult:
        """Analyze a local frame file and return Nova response."""
        image_path = Path(frame_path)
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Frame path does not exist: {frame_path}")

        frame_bytes = image_path.read_bytes()
        return await self._screen_analysis.analyze_screen_frame(
            session_id=session_id,
            user_prompt=prompt,
            frame_bytes=frame_bytes,
        )

    async def ingest_screen_frame(
        self,
        session_id: str,
        frame_bytes: bytes,
        source_type: str = "screen_share",
        analysis_prompt: str | None = None,
        force_process: bool = False,
    ) -> FrameIngestionResult:
        """Sample, OCR, and index a screen frame for retrieval memory."""
        if not frame_bytes:
            raise ValueError("Frame bytes are required for ingestion")

        if not force_process and not self._screen_analysis.should_process_frame(session_id):
            now = datetime.now(timezone.utc)
            previous = self._session_frames.get(session_id)
            return FrameIngestionResult(
                session_id=session_id,
                processed=False,
                skipped_reason="capture_interval_not_elapsed",
                ocr_text=previous.ocr_text if previous else "",
                ocr_record_id=None,
                analysis_result=None,
                ingested_at=now,
            )

        ingested_at = datetime.now(timezone.utc)
        ocr_text = (await asyncio.to_thread(self._nova_client.extract_text_from_image, frame_bytes)).strip()

        ocr_record_id: str | None = None
        if ocr_text:
            ocr_record_id = await self._memory.index_content(
                content=ocr_text,
                source_type=f"{source_type}_ocr",
                metadata={
                    "session_id": session_id,
                    "captured_at": ingested_at.isoformat(),
                    "source_type": source_type,
                },
            )

        self._session_frames[session_id] = SessionFrameSnapshot(
            frame_bytes=frame_bytes,
            ocr_text=ocr_text,
            source_type=source_type,
            ingested_at=ingested_at,
        )

        analysis_result: ScreenAnalysisResult | None = None
        if analysis_prompt and analysis_prompt.strip():
            analysis_result = await self._screen_analysis.analyze_screen_frame(
                session_id=session_id,
                user_prompt=analysis_prompt,
                frame_bytes=frame_bytes,
                ocr_text=ocr_text,
            )

        return FrameIngestionResult(
            session_id=session_id,
            processed=True,
            skipped_reason=None,
            ocr_text=ocr_text,
            ocr_record_id=ocr_record_id,
            analysis_result=analysis_result,
            ingested_at=ingested_at,
        )

    async def answer_with_session_context(self, session_id: str, prompt: str) -> ScreenAnalysisResult:
        """Answer a user prompt using the latest ingested session frame context."""
        snapshot = self._session_frames.get(session_id)
        frame_bytes = snapshot.frame_bytes if snapshot else None
        ocr_text = snapshot.ocr_text if snapshot else None

        return await self._screen_analysis.analyze_screen_frame(
            session_id=session_id,
            user_prompt=prompt,
            frame_bytes=frame_bytes,
            ocr_text=ocr_text,
        )

    def get_session_snapshot(self, session_id: str) -> SessionFrameSnapshot | None:
        """Return latest frame snapshot for a session if available."""
        return self._session_frames.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear in-memory frame state for a session."""
        self._session_frames.pop(session_id, None)

    async def generate_overlay_diagram(self, prompt: str) -> str:
        """Generate Mermaid overlay diagram text from a user request."""
        context = await self._memory.retrieve(prompt)
        return await self._screen_analysis.generate_overlay_mermaid(prompt, context)

    @property
    def config(self) -> AssistantConfig:
        """Expose read-only runtime config."""
        return self._config
