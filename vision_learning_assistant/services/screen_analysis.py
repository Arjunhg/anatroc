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

    def clear_session(self, session_id: str) -> None:
        """Reset per-session frame throttling state."""
        self._last_processed_by_session.pop(session_id, None)

    async def analyze_screen_frame(
        self,
        session_id: str,
        session_run_id: str | None,
        user_prompt: str,
        frame_bytes: bytes | None,
        ocr_text: str | None = None,
    ) -> ScreenAnalysisResult:
        """Analyze a sampled screen frame and store resulting context."""
        normalized_ocr = (ocr_text or "").strip()
        retrieved = []
        if not (_is_direct_screen_question(user_prompt) and normalized_ocr):
            retrieved = await self._memory_service.retrieve_for_session(
                question=user_prompt,
                session_id=session_id,
                session_run_id=session_run_id,
                source_types=["screen_share_ocr", "camera_ocr"],
            )
        context_payload = [
            item.content
            for item in retrieved
            if item.content.strip() and item.content.strip() != normalized_ocr
        ]

        response_text = await asyncio.to_thread(
            self._nova_client.analyze_screen,
            user_prompt,
            frame_bytes,
            normalized_ocr,
            context_payload,
        )

        return ScreenAnalysisResult(
            response_text=response_text,
            context_hits=len(retrieved),
            context_ids=[item.id for item in retrieved],
            stored_record_id=None,
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


def _is_direct_screen_question(prompt: str) -> bool:
    """Detect direct questions about what is visible on screen right now."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(
        hint in normalized
        for hint in (
            "what am i reading",
            "what i am reading",
            "what i'm reading",
            "what am i seeing",
            "what i am seeing",
            "what's on the screen",
            "what is on the screen",
            "on the screen right now",
            "what do you see on the screen",
            "tell me what i'm reading",
            "tell me what i am reading",
        )
    )
