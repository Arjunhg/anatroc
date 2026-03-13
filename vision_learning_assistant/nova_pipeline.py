"""Nova-first orchestration pipeline for screen analysis and retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

from vision_learning_assistant.config import AssistantConfig
from vision_learning_assistant.nova.bedrock_client import NovaBedrockClient
from vision_learning_assistant.services.context_memory import ContextMemoryService
from vision_learning_assistant.services.screen_analysis import ScreenAnalysisResult, ScreenAnalysisService
from vision_learning_assistant.storage.aurora_store import AuroraVectorStore
from vision_learning_assistant.storage.redis_cache import RedisCache

LOGGER = logging.getLogger(__name__)


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

    async def generate_overlay_diagram(self, prompt: str) -> str:
        """Generate Mermaid overlay diagram text from a user request."""
        context = await self._memory.retrieve(prompt)
        return await self._screen_analysis.generate_overlay_mermaid(prompt, context)

    @property
    def config(self) -> AssistantConfig:
        """Expose read-only runtime config."""
        return self._config
