"""Screen-share analysis workflow using Nova Lite and context memory."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Sequence

from vision_learning_assistant.nova.bedrock_client import NovaBedrockClient
from vision_learning_assistant.services.context_memory import ContextMemoryService
from vision_learning_assistant.storage.aurora_store import RetrievedContext


@dataclass(slots=True)
class ScreenAnalysisResult:
    """Structured result from a single screen analysis request."""

    response_text: str
    context_hits: int
    context_ids: list[str]
    stored_record_id: str | None


class ScreenAnalysisService:
    """Coordinates screen analysis, retrieval, and memory indexing."""

    def __init__(
        self,
        nova_client: NovaBedrockClient,
        memory_service: ContextMemoryService,
        capture_interval_seconds: float,
    ) -> None:
        """Initialize analysis dependencies and frame sampling controls."""
        self._nova_client = nova_client
        self._memory_service = memory_service
        self._capture_interval_seconds = max(1.0, capture_interval_seconds)
        self._last_processed_by_session: dict[str, float] = {}

    def should_process_frame(self, session_id: str) -> bool:
        """Return True when enough time passed since last frame for this session."""
        now = time.monotonic()
        last_processed = self._last_processed_by_session.get(session_id)
        if last_processed is None:
            self._last_processed_by_session[session_id] = now
            return True
        if (now - last_processed) >= self._capture_interval_seconds:
            self._last_processed_by_session[session_id] = now
            return True
        return False

    async def analyze_screen_frame(
        self,
        session_id: str,
        user_prompt: str,
        frame_bytes: bytes | None,
        ocr_text: str | None = None,
    ) -> ScreenAnalysisResult:
        """Analyze a sampled screen frame and store resulting context."""
        retrieved = await self._memory_service.retrieve(user_prompt)
        context_payload = [item.content for item in retrieved]

        response_text = await asyncio.to_thread(
            self._nova_client.analyze_screen,
            user_prompt,
            frame_bytes,
            context_payload,
        )

        stored_record_id = await self._memory_service.index_content(
            content=ocr_text.strip() if ocr_text and ocr_text.strip() else response_text,
            source_type="screen_analysis",
            metadata={
                "session_id": session_id,
                "prompt": user_prompt,
            },
        )

        return ScreenAnalysisResult(
            response_text=response_text,
            context_hits=len(retrieved),
            context_ids=[item.id for item in retrieved],
            stored_record_id=stored_record_id,
        )

    async def generate_overlay_mermaid(
        self,
        prompt: str,
        retrieved_context: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Generate Mermaid diagram text for overlay rendering."""
        context_payload = [item.content for item in (retrieved_context or [])]
        return await asyncio.to_thread(
            self._nova_client.generate_mermaid_diagram,
            prompt,
            context_payload,
        )
