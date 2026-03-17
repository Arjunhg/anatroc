"""Nova API server for Stream token issuance and live pipeline ingestion.

Run with:
    uv run python token_server.py

This server provides:
- Stream Video JWT tokens for frontend call setup
- session lifecycle endpoints for the Nova pipeline
- sampled screen frame ingestion with Nova OCR + indexing
- synchronized user query endpoint that uses latest ingested frame context
"""

from __future__ import annotations

import base64
import re
import asyncio
import contextlib
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vision_learning_assistant import AssistantConfig, VisionLearningPipeline
from vision_learning_assistant.storage.aurora_store import AuroraVectorStore
from vision_learning_assistant.nova import DEFAULT_SONIC_SYSTEM_PROMPT, NovaSonicWebSocketSession
from vision_learning_assistant.nova_pipeline import FrameIngestionResult
from vision_learning_assistant.services.screen_analysis import ScreenAnalysisResult

load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

LOGGER = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

STREAM_API_KEY: str = os.getenv("STREAM_API_KEY", "").strip()
STREAM_API_SECRET: str = os.getenv("STREAM_API_SECRET", "").strip()
TOKEN_TTL_SECONDS: int = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))
HOST: str = os.getenv("TOKEN_SERVER_HOST", "127.0.0.1")
PORT: int = int(os.getenv("TOKEN_SERVER_PORT", "8001"))
CORS_ALLOW_ORIGINS: str = os.getenv("TOKEN_SERVER_CORS_ALLOW_ORIGINS", "*")
CORS_ALLOW_HEADERS: str = os.getenv(
    "TOKEN_SERVER_CORS_ALLOW_HEADERS",
    "Content-Type,Authorization",
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_FRONTEND_CANDIDATES = (
    _PROJECT_ROOT / "frontend1" / "dist",
    _PROJECT_ROOT / "frontend",
)
_FRONTEND_DIR = next((path for path in _FRONTEND_CANDIDATES if path.exists()), _FRONTEND_CANDIDATES[0])
_ALLOWED_ORIGINS = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]
_ALLOWED_HEADERS = [header.strip() for header in CORS_ALLOW_HEADERS.split(",") if header.strip()]


@dataclass(slots=True)
class SessionRuntimeState:
    """In-memory runtime state for one frontend session."""

    session_id: str
    run_id: str
    created_at: datetime
    frame_ingest_count: int = 0
    last_ingested_at: datetime | None = None
    last_ocr_text: str = ""
    last_response_text: str = ""
    overlay_mermaid: str = ""
    overlay_prompt: str = ""
    overlay_updated_at: datetime | None = None
    overlay_position: str = "top-left"
    overlay_minimized: bool = False
    last_sonic_ocr_push_at: float = 0.0
    pending_sonic_ocr: str = ""
    sonic_assistant_speaking: bool = False


@dataclass(slots=True)
class SonicBridgeState:
    """Transient per-session Sonic state used for retrieval-grounded turns."""

    current_user_transcript_parts: list[str] = field(default_factory=list)
    current_assistant_transcript_parts: list[str] = field(default_factory=list)
    last_user_transcript: str = ""
    last_context_payload: str = ""
    last_retrieval_at_monotonic: float = 0.0
    user_turn_count: int = 0
    assistant_turn_count: int = 0


class StartSessionRequest(BaseModel):
    """Payload for explicit session start."""

    session_id: str | None = None


class StopSessionRequest(BaseModel):
    """Payload for explicit session stop."""

    session_id: str


class FrameIngestRequest(BaseModel):
    """Payload containing sampled frame bytes from screen share."""

    session_id: str
    image_base64: str
    source_type: str = "screen_share"
    analysis_prompt: str | None = None
    force_process: bool = False


class VoiceQueryRequest(BaseModel):
    """Payload for synchronized voice/text reasoning turns."""

    session_id: str
    user_text: str = Field(min_length=1)


class DiagramRequest(BaseModel):
    """Payload for overlay diagram generation."""

    prompt: str = Field(min_length=1)


class LegacyStartAgentRequest(BaseModel):
    """Backwards-compatible payload shape for older frontend clients."""

    call_id: str = "vision-session-1"
    call_type: str = "default"


class LegacyStopAgentRequest(BaseModel):
    """Backwards-compatible payload shape for older frontend clients."""

    call_id: str | None = None


# Aurora maintenance helpers
def run_vacuum_analyze(pool):
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("VACUUM ANALYZE embeddings;")
        conn.commit()

def schedule_aurora_maintenance(aurora_store: AuroraVectorStore):
    import asyncio
    from datetime import datetime
    async def maintenance_loop():
        while True:
            try:
                await aurora_store.cleanup_embeddings()
                # Run VACUUM ANALYZE every 4 hours
                if int(datetime.now().hour) % 4 == 0:
                    await aurora_store.vacuum_analyze_embeddings()
            except Exception as exc:
                LOGGER.warning("Aurora maintenance failed: %s", exc)
            await asyncio.sleep(3600)  # Run every hour
    asyncio.create_task(maintenance_loop())

app = FastAPI(title="Anatroc Assistant API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=_ALLOWED_HEADERS or ["*"],
)

if _FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")
if (_FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIR / "assets")), name="frontend-assets")

_pipeline: VisionLearningPipeline | None = None
_sessions: dict[str, SessionRuntimeState] = {}
_sonic_sessions: dict[str, NovaSonicWebSocketSession] = {}
_sonic_websockets: dict[str, WebSocket] = {}
_sonic_bridge_state: dict[str, SonicBridgeState] = {}

_SONIC_RETRIEVAL_HINTS = (
    "start",
    "earlier",
    "before",
    "previous",
    "history",
    "summary",
    "summarize",
    "recap",
    "flow",
    "timeline",
    "done",
    "doing",
    "happening",
    "where",
    "initial",
    "initially",
    "read",
    "reading",
    "screen",
    "blog",
    "article",
)

_OVERLAY_GENERATE_HINTS = (
    "diagram",
    "architecture",
    "flowchart",
    "sequence",
    "workflow",
    "mermaid",
)

_OVERLAY_CLEAR_HINTS = (
    "remove overlay",
    "remove the overlay",
    "remove this overlay",
    "remove diagram",
    "remove the diagram",
    "remove this diagram",
    "delete overlay",
    "delete the overlay",
    "delete diagram",
    "delete the diagram",
    "clear overlay",
    "clear the overlay",
    "clear diagram",
    "clear the diagram",
    "hide overlay",
    "hide the overlay",
    "hide diagram",
    "hide the diagram",
    "close overlay",
    "close the overlay",
    "close diagram",
    "close the diagram",
    "get rid of the overlay",
    "get rid of the diagram",
)

_OVERLAY_EXPORT_HINTS = (
    "export overlay",
    "export the overlay",
    "export diagram",
    "export the diagram",
    "export this diagram",
    "download overlay",
    "download the overlay",
    "download diagram",
    "download the diagram",
    "copy overlay",
    "copy the overlay",
    "copy diagram",
    "copy the diagram",
    "save diagram",
    "save the diagram",
    "save overlay",
    "save the overlay",
)

_OVERLAY_MINIMIZE_HINTS = (
    "minimize overlay",
    "minimize the overlay",
    "minimize diagram",
    "minimize the diagram",
    "collapse overlay",
    "collapse the overlay",
    "collapse diagram",
    "collapse the diagram",
    "shrink overlay",
    "shrink the overlay",
    "shrink diagram",
    "shrink the diagram",
    "make it smaller",
    "make the diagram smaller",
    "make the overlay smaller",
)

_OVERLAY_EXPAND_HINTS = (
    "expand overlay",
    "expand the overlay",
    "expand diagram",
    "expand the diagram",
    "maximize overlay",
    "maximize the overlay",
    "maximize diagram",
    "maximize the diagram",
    "open overlay",
    "open the overlay",
    "open diagram",
    "open the diagram",
    "restore overlay",
    "restore the overlay",
    "restore diagram",
    "restore the diagram",
    "make it bigger",
    "make the diagram bigger",
    "make the overlay bigger",
    "show the diagram",
    "show overlay",
    "show the overlay",
)

_SONIC_RUNTIME_SYSTEM_PROMPT = (
    DEFAULT_SONIC_SYSTEM_PROMPT
    + " You have access to the user's screen through OCR text updates."
    + " These updates arrive as messages starting with '[SCREEN UPDATE]'."
    + " When you receive a screen update, briefly note what is on screen in 1-2 short sentences."
    + " Do NOT repeat the same description if the screen content has not changed."
    + " When the user asks you a question about the screen, answer in full detail."
    + " You CAN see the screen — NEVER say you cannot see it or need an update."
    + " NEVER ask the user to share their screen or provide text — you already have it."
    + " You CAN generate architecture diagrams, flow diagrams, and visual overviews."
    + " When the user asks for a diagram, say 'I will generate that diagram for you now' and describe what it will show."
    + " NEVER say you cannot create or display diagrams — the system handles diagram rendering automatically."
    + " You can also control the diagram overlay: export it, minimize it, expand it, move it, or remove it."
    + " When the user asks to export, minimize, move, or remove the diagram, confirm the action briefly — e.g. 'Done, I have exported the diagram.'"
)


def _generate_token(user_id: str) -> str:
    """Generate a Stream Video JWT for the given user ID."""
    if not STREAM_API_SECRET:
        raise RuntimeError("STREAM_API_SECRET is missing. Set Stream credentials in .env.")
    now = int(time.time())
    payload: dict[str, object] = {
        "user_id": user_id,
        "iss": "stream-video-python",
        "sub": f"user/{user_id}",
        "iat": now - 5,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, STREAM_API_SECRET, algorithm="HS256")


def _decode_image_bytes(image_base64: str) -> bytes:
    """Decode base64 payload from raw/base64-data-url image string."""
    encoded = image_base64.strip()
    if not encoded:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {exc}") from exc


def _require_pipeline() -> VisionLearningPipeline:
    """Return initialized pipeline or fail fast with 503."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def _ensure_session(session_id: str) -> SessionRuntimeState:
    """Get or create in-memory session state."""
    normalized = session_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="session_id is required")

    existing = _sessions.get(normalized)
    if existing:
        return existing

    state = SessionRuntimeState(
        session_id=normalized,
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
    )
    _sessions[normalized] = state
    return state


def _serialize_screen_result(result: ScreenAnalysisResult) -> dict[str, Any]:
    """Convert ScreenAnalysisResult into API response payload."""
    return {
        "response_text": result.response_text,
        "context_hits": result.context_hits,
        "context_ids": result.context_ids,
        "stored_record_id": result.stored_record_id,
    }


def _serialize_frame_ingestion(result: FrameIngestionResult) -> dict[str, Any]:
    """Convert FrameIngestionResult into API response payload."""
    payload: dict[str, Any] = {
        "session_id": result.session_id,
        "processed": result.processed,
        "skipped_reason": result.skipped_reason,
        "ocr_text": result.ocr_text,
        "ocr_record_id": result.ocr_record_id,
        "ingested_at": result.ingested_at.isoformat(),
    }
    if result.analysis_result is not None:
        payload["analysis"] = _serialize_screen_result(result.analysis_result)
    else:
        payload["analysis"] = None
    return payload


def _append_transcript_chunk(chunks: list[str], text: str) -> None:
    """Append a transcript chunk while avoiding immediate duplicates."""
    normalized = text.strip()
    if not normalized:
        return
    if chunks and chunks[-1] == normalized:
        return
    chunks.append(normalized)


def _collapse_transcript_chunks(chunks: list[str]) -> str:
    """Collapse transcript chunks into one readable sentence."""
    collapsed = " ".join(chunk.strip() for chunk in chunks if chunk.strip())
    return re.sub(r"\s+", " ", collapsed).strip()


def _clip_text(text: str, max_chars: int) -> str:
    """Clip long text blocks to a bounded size for Sonic context injection."""
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _should_retrieve_sonic_context(prompt: str) -> bool:
    """Use a lightweight heuristic to limit expensive retrievals to memory-seeking turns."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _SONIC_RETRIEVAL_HINTS)


def _should_generate_overlay(prompt: str) -> bool:
    """Detect whether the user is asking for a visual flow/diagram overlay."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    if any(hint in normalized for hint in _OVERLAY_CLEAR_HINTS):
        return False
    if not any(hint in normalized for hint in _OVERLAY_GENERATE_HINTS):
        return False
    if any(
        verb in normalized
        for verb in ("generate", "draw", "show", "create", "build", "make", "overlay", "provide", "give")
    ):
        return True
    return "overlay diagram" in normalized or "architecture diagram" in normalized


def _should_force_context_refresh(prompt: str) -> bool:
    """Force OCR context push for direct screen-reading prompts even when text is unchanged."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(
        hint in normalized
        for hint in (
            "what am i reading",
            "what i'm reading",
            "what i am reading",
            "on the screen",
            "screen right now",
            "reading right now",
            "this article",
            "this blog",
            "this flow",
        )
    )


def _is_screen_content_question(prompt: str) -> bool:
    """Detect prompts asking what is currently visible/read on screen."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(
        hint in normalized
        for hint in (
            "what am i reading",
            "what i am reading",
            "what i'm reading",
            "reading about",
            "what am i seeing",
            "what i am seeing",
            "what's on the screen",
            "what is on the screen",
            "on the screen",
            "screen right now",
            "currently reading",
            "explain what i am reading",
            "explain this",
            "summarize what i have been reading",
        )
    )


def _should_clear_overlay(prompt: str) -> bool:
    """Detect voice commands that remove the active overlay."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(hint in normalized for hint in _OVERLAY_CLEAR_HINTS)


def _should_export_overlay(prompt: str) -> bool:
    """Detect voice commands that request Mermaid export/copy."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(hint in normalized for hint in _OVERLAY_EXPORT_HINTS)


def _should_minimize_overlay(prompt: str) -> bool:
    """Detect voice commands that minimize overlay card."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(hint in normalized for hint in _OVERLAY_MINIMIZE_HINTS)


def _should_expand_overlay(prompt: str) -> bool:
    """Detect voice commands that restore expanded overlay card."""
    normalized = prompt.lower().strip()
    if not normalized:
        return False
    return any(hint in normalized for hint in _OVERLAY_EXPAND_HINTS)


def _extract_overlay_position(prompt: str) -> str | None:
    """Infer target overlay position from voice command."""
    normalized = prompt.lower().strip()
    if "overlay" not in normalized and "diagram" not in normalized:
        return None
    if "top left" in normalized or "upper left" in normalized:
        return "top-left"
    if "top right" in normalized or "upper right" in normalized:
        return "top-right"
    if "bottom left" in normalized or "lower left" in normalized:
        return "bottom-left"
    if "bottom right" in normalized or "lower right" in normalized:
        return "bottom-right"
    if "center" in normalized or "middle" in normalized:
        return "center"
    if "left" in normalized:
        return "top-left"
    if "right" in normalized:
        return "top-right"
    if "bottom" in normalized:
        return "bottom-left"
    if "top" in normalized:
        return "top-left"
    return None


async def _persist_sonic_memory(
    pipeline: VisionLearningPipeline,
    session_id: str,
    session_run_id: str,
    content: str,
    source_type: str,
    turn_index: int,
) -> str | None:
    """Persist a Sonic turn into Aurora-backed memory when text is available."""
    normalized = content.strip()
    if not normalized:
        return None
    record_id = await pipeline.index_session_memory(
        content=normalized,
        source_type=source_type,
        metadata={
            "session_id": session_id,
            "session_run_id": session_run_id,
            "turn_index": turn_index,
            "channel": "sonic_websocket",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    if record_id:
        LOGGER.info(
            "Persisted Sonic memory session=%s source_type=%s turn=%d record_id=%s",
            session_id,
            source_type,
            turn_index,
            record_id,
        )
    return record_id


async def _handle_sonic_overlay_intent(
    pipeline: VisionLearningPipeline,
    session_state: SessionRuntimeState,
    user_prompt: str,
    websocket: WebSocket,
) -> None:
    """Generate or clear session overlay diagrams based on a voice turn."""
    normalized_prompt = user_prompt.strip()
    if not normalized_prompt:
        return

    if _should_clear_overlay(normalized_prompt):
        session_state.overlay_mermaid = ""
        session_state.overlay_prompt = ""
        session_state.overlay_updated_at = datetime.now(timezone.utc)
        session_state.overlay_minimized = False
        await websocket.send_json(
            {
                "type": "overlay_clear",
                "source": "voice_intent",
            }
        )
        LOGGER.info("Cleared overlay for session=%s from voice intent", session_state.session_id)
        return

    if session_state.overlay_mermaid.strip():
        requested_position = _extract_overlay_position(normalized_prompt)
        if requested_position and requested_position != session_state.overlay_position:
            session_state.overlay_position = requested_position
            await websocket.send_json(
                {
                    "type": "overlay_control",
                    "action": "move",
                    "position": requested_position,
                    "source": "voice_intent",
                }
            )
            LOGGER.info("Moved overlay for session=%s to %s", session_state.session_id, requested_position)
            return

        if _should_minimize_overlay(normalized_prompt):
            session_state.overlay_minimized = True
            await websocket.send_json(
                {
                    "type": "overlay_control",
                    "action": "minimize",
                    "source": "voice_intent",
                }
            )
            return

        if _should_expand_overlay(normalized_prompt):
            session_state.overlay_minimized = False
            await websocket.send_json(
                {
                    "type": "overlay_control",
                    "action": "expand",
                    "source": "voice_intent",
                }
            )
            return

        if _should_export_overlay(normalized_prompt):
            await websocket.send_json(
                {
                    "type": "overlay_export",
                    "mermaid": session_state.overlay_mermaid,
                    "prompt": session_state.overlay_prompt,
                    "session_id": session_state.session_id,
                    "source": "voice_intent",
                }
            )
            return

    if not _should_generate_overlay(normalized_prompt):
        return

    try:
        mermaid = await pipeline.generate_overlay_diagram_for_session(
            session_state.session_id,
            normalized_prompt,
            session_run_id=session_state.run_id,
        )
    except Exception as exc:
        LOGGER.warning("Overlay diagram generation failed for session=%s: %s", session_state.session_id, exc)
        await websocket.send_json(
            {
                "type": "overlay_error",
                "message": str(exc),
            }
        )
        return

    normalized_mermaid = mermaid.strip()
    if not normalized_mermaid:
        return

    # Strip markdown fencing if present — mermaid.render() needs raw syntax
    normalized_mermaid = re.sub(r'^```(?:mermaid)?\s*\n?', '', normalized_mermaid)
    normalized_mermaid = re.sub(r'\n?```\s*$', '', normalized_mermaid)
    normalized_mermaid = normalized_mermaid.strip()
    if not normalized_mermaid:
        return

    session_state.overlay_mermaid = normalized_mermaid
    session_state.overlay_prompt = normalized_prompt
    session_state.overlay_updated_at = datetime.now(timezone.utc)
    session_state.overlay_minimized = False
    await websocket.send_json(
        {
            "type": "overlay_diagram",
            "mermaid": normalized_mermaid,
            "prompt": normalized_prompt,
            "source": "voice_intent",
            "updated_at": session_state.overlay_updated_at.isoformat(),
            "position": session_state.overlay_position,
            "minimized": session_state.overlay_minimized,
        }
    )
    LOGGER.info("Generated overlay diagram for session=%s", session_state.session_id)


async def _build_sonic_context_payload(
    pipeline: VisionLearningPipeline,
    session_id: str,
    session_run_id: str,
    user_prompt: str,
    bridge_state: SonicBridgeState,
) -> tuple[str, int]:
    """Build a compact retrieval-grounded context payload for Sonic."""
    config = pipeline.config
    sections: list[str] = []
    used_chars = 0

    normalized_prompt = user_prompt.strip()

    if normalized_prompt:
        sections.append(f"[User question]\n{_clip_text(normalized_prompt, 240)}")

    snapshot = pipeline.get_session_snapshot(session_id)
    latest_ocr = snapshot.ocr_text.strip() if snapshot and snapshot.ocr_text.strip() else ""
    direct_screen_question = _is_screen_content_question(user_prompt)
    if latest_ocr:
        sections.append(
            "[Grounding rule]\nUse the OCR below as the source of truth for what is currently on screen."
        )
        ocr_block = _clip_text(latest_ocr, min(config.sonic_context_max_total_chars, config.sonic_context_max_chars_per_hit))
        sections.append(f"[Current screen OCR]\n{ocr_block}")
        used_chars += len(ocr_block)

    retrieved_count = 0
    now_monotonic = time.monotonic()
    retrieval_allowed = (
        config.enable_sonic_context_retrieval
        and normalized_prompt
        and _should_retrieve_sonic_context(user_prompt)
        and not (direct_screen_question and latest_ocr)
        and (
            bridge_state.last_retrieval_at_monotonic <= 0.0
            or now_monotonic - bridge_state.last_retrieval_at_monotonic
            >= config.sonic_context_retrieval_cooldown_seconds
        )
    )
    if retrieval_allowed:
        retrieval_source_types = ["screen_share_ocr", "camera_ocr"]
        if not direct_screen_question:
            retrieval_source_types.extend(["sonic_user_turn", "sonic_assistant_turn"])
        retrieved = await pipeline.retrieve_context_memory(
            user_prompt,
            limit=config.sonic_context_max_results,
            session_id=session_id,
            session_run_id=session_run_id,
            source_types=retrieval_source_types,
        )
        bridge_state.last_retrieval_at_monotonic = now_monotonic
        seen_contents = {latest_ocr}
        memory_lines: list[str] = []
        for item in retrieved:
            clipped = _clip_text(item.content, config.sonic_context_max_chars_per_hit)
            if not clipped or clipped in seen_contents:
                continue
            projected_chars = used_chars + len(clipped)
            if projected_chars > config.sonic_context_max_total_chars:
                break
            seen_contents.add(clipped)
            memory_lines.append(f"- ({item.source_type}, score={item.score:.2f}) {clipped}")
            used_chars = projected_chars
        if memory_lines:
            sections.append("[Retrieved memory]\n" + "\n".join(memory_lines))
            retrieved_count = len(memory_lines)

    return "\n\n".join(section for section in sections if section.strip()), retrieved_count


async def _stop_sonic_session(session_id: str) -> None:
    """Close and remove active Sonic session bridge for a session."""
    sonic = _sonic_sessions.pop(session_id, None)
    if sonic is None:
        return
    try:
        await sonic.close()
    except Exception as exc:
        LOGGER.warning("Failed to close Sonic session %s: %s", session_id, exc)


async def _stop_sonic_websocket(session_id: str) -> None:
    """Close and remove active Sonic websocket for a session."""
    websocket = _sonic_websockets.pop(session_id, None)
    _sonic_bridge_state.pop(session_id, None)
    if websocket is None:
        return
    with contextlib.suppress(Exception):
        await websocket.close(code=1000, reason="Session stopped")



@app.on_event("startup")
async def _startup() -> None:
    """Initialize Nova pipeline once on service startup."""
    global _pipeline
    config = AssistantConfig.from_env()
    _pipeline = VisionLearningPipeline(config)
    await _pipeline.start()
    # Schedule Aurora maintenance if AuroraVectorStore is present
    aurora_store = None
    if hasattr(_pipeline, '_memory') and hasattr(_pipeline._memory, '_vector_store'):
        aurora_store = _pipeline._memory._vector_store
    if isinstance(aurora_store, AuroraVectorStore):
        schedule_aurora_maintenance(aurora_store)
    elif aurora_store is not None:
        LOGGER.warning("Aurora maintenance scheduling skipped: unexpected vector store type %s", type(aurora_store))
    LOGGER.info("Nova API server started with mode=%s", config.assistant_mode)


@app.on_event("shutdown")
async def _shutdown() -> None:
    """Close pipeline resources on service shutdown."""
    global _pipeline
    for sonic in list(_sonic_sessions.values()):
        try:
            await sonic.close()
        except Exception:
            LOGGER.exception("Failed to close Sonic bridge during shutdown")
    _sonic_sessions.clear()
    for websocket in list(_sonic_websockets.values()):
        with contextlib.suppress(Exception):
            await websocket.close(code=1001, reason="Server shutdown")
    _sonic_websockets.clear()

    if _pipeline is not None:
        await _pipeline.close()
    _pipeline = None
    _sessions.clear()
    LOGGER.info("Nova API server shutdown complete")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check for server and pipeline runtime."""
    return {
        "status": "ok",
        "service": "vision-nova-api",
        "pipeline_ready": _pipeline is not None,
        "active_sessions": len(_sessions),
    }


@app.get("/token")
async def token(user_id: str = Query(min_length=1)) -> dict[str, str]:
    """Issue Stream Video JWT token for a user."""
    if not STREAM_API_KEY:
        raise HTTPException(status_code=500, detail="STREAM_API_KEY is missing")
    try:
        issued = _generate_token(user_id=user_id.strip())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"token": issued, "api_key": STREAM_API_KEY}


@app.post("/api/session/start")
async def start_session(payload: StartSessionRequest) -> dict[str, Any]:
    """Start a fresh pipeline session."""
    session_id = (payload.session_id or str(uuid.uuid4())).strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if _pipeline is not None:
        _pipeline.clear_session(session_id)
    await _stop_sonic_session(session_id)
    await _stop_sonic_websocket(session_id)
    _sessions[session_id] = SessionRuntimeState(
        session_id=session_id,
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
    )
    return {"status": "started", "session_id": session_id}


@app.post("/api/session/stop")
async def stop_session(payload: StopSessionRequest) -> dict[str, Any]:
    """Stop a session and clear associated in-memory frame state."""
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    pipeline = _require_pipeline()
    pipeline.clear_session(session_id)
    await _stop_sonic_session(session_id)
    await _stop_sonic_websocket(session_id)
    _sessions.pop(session_id, None)
    return {"status": "stopped", "session_id": session_id}


@app.get("/api/session/{session_id}")
async def session_state(session_id: str) -> dict[str, Any]:
    """Return current session runtime metadata."""
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    pipeline = _require_pipeline()
    snapshot = pipeline.get_session_snapshot(session_id)
    return {
        "session_id": state.session_id,
        "created_at": state.created_at.isoformat(),
        "frame_ingest_count": state.frame_ingest_count,
        "last_ingested_at": state.last_ingested_at.isoformat() if state.last_ingested_at else None,
        "last_ocr_text": state.last_ocr_text,
        "last_response_text": state.last_response_text,
        "overlay_prompt": state.overlay_prompt,
        "overlay_mermaid": state.overlay_mermaid,
        "overlay_updated_at": state.overlay_updated_at.isoformat() if state.overlay_updated_at else None,
        "overlay_position": state.overlay_position,
        "overlay_minimized": state.overlay_minimized,
        "has_latest_frame": snapshot is not None,
        "latest_frame_source_type": snapshot.source_type if snapshot else None,
        "latest_frame_ingested_at": snapshot.ingested_at.isoformat() if snapshot else None,
    }


@app.post("/api/frame/ingest")
async def ingest_frame(payload: FrameIngestRequest) -> dict[str, Any]:
    """Ingest one sampled screen frame, run OCR, and index memory."""
    pipeline = _require_pipeline()
    state = _ensure_session(payload.session_id)
    frame_bytes = _decode_image_bytes(payload.image_base64)

    result = await pipeline.ingest_screen_frame(
        session_id=state.session_id,
        session_run_id=state.run_id,
        frame_bytes=frame_bytes,
        source_type=payload.source_type,
        analysis_prompt=payload.analysis_prompt,
        force_process=payload.force_process,
    )
    if result.processed:
        state.frame_ingest_count += 1
        state.last_ingested_at = result.ingested_at
        state.last_ocr_text = result.ocr_text

        # Queue OCR for Sonic — only push if assistant is NOT speaking AND cooldown elapsed
        sonic = _sonic_sessions.get(state.session_id)
        if sonic and result.ocr_text.strip():
            state.pending_sonic_ocr = _clip_text(result.ocr_text, 4000)
            now_mono = time.monotonic()
            cooldown_ok = now_mono - state.last_sonic_ocr_push_at >= 30.0
            if not state.sonic_assistant_speaking and cooldown_ok:
                try:
                    await sonic.send_context_update(state.pending_sonic_ocr)
                    state.pending_sonic_ocr = ""
                    state.last_sonic_ocr_push_at = now_mono
                    LOGGER.info(
                        "Pushed ingest OCR to Sonic session=%s ocr_len=%d",
                        state.session_id,
                        len(result.ocr_text),
                    )
                except Exception as exc:
                    LOGGER.warning("Failed pushing ingest OCR to Sonic: %s", exc)
            else:
                LOGGER.debug(
                    "Queued OCR for Sonic (speaking=%s cooldown=%s) session=%s",
                    state.sonic_assistant_speaking,
                    not cooldown_ok,
                    state.session_id,
                )

        if result.analysis_result is not None:
            state.last_response_text = result.analysis_result.response_text

    return _serialize_frame_ingestion(result)


@app.post("/api/voice/query")
async def voice_query(payload: VoiceQueryRequest) -> dict[str, Any]:
    """Answer a user turn using latest synchronized frame context."""
    pipeline = _require_pipeline()
    state = _ensure_session(payload.session_id)
    prompt = payload.user_text.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="user_text must not be blank")

    result = await pipeline.answer_with_session_context(
        session_id=state.session_id,
        session_run_id=state.run_id,
        prompt=prompt,
    )
    state.last_response_text = result.response_text
    return _serialize_screen_result(result)


@app.post("/api/diagram")
async def generate_diagram(payload: DiagramRequest) -> dict[str, str]:
    """Generate Mermaid overlay text from a prompt."""
    pipeline = _require_pipeline()
    mermaid = await pipeline.generate_overlay_diagram(payload.prompt.strip())
    return {"mermaid": mermaid}


@app.post("/start-agent")
async def legacy_start_agent(payload: LegacyStartAgentRequest) -> dict[str, Any]:
    """Backwards-compatible session start endpoint used by old clients."""
    session_id = payload.call_id.strip() or "vision-session-1"
    _ensure_session(session_id)
    return {
        "status": "started",
        "session_id": session_id,
        "call_type": payload.call_type,
        "message": "Legacy start-agent mapped to /api/session/start",
    }


@app.post("/stop-agent")
async def legacy_stop_agent(payload: LegacyStopAgentRequest) -> dict[str, Any]:
    """Backwards-compatible session stop endpoint used by old clients."""
    pipeline = _require_pipeline()
    if payload.call_id:
        session_id = payload.call_id.strip()
        pipeline.clear_session(session_id)
        await _stop_sonic_session(session_id)
        await _stop_sonic_websocket(session_id)
        _sessions.pop(session_id, None)
        return {"status": "stopped", "session_id": session_id}

    for session_id in list(_sessions.keys()):
        pipeline.clear_session(session_id)
        await _stop_sonic_session(session_id)
        await _stop_sonic_websocket(session_id)
        _sessions.pop(session_id, None)
    return {"status": "stopped_all"}


@app.websocket("/ws/sonic/{session_id}")
async def sonic_ws(websocket: WebSocket, session_id: str) -> None:
    """Bridge browser microphone audio to Nova Sonic and stream audio/text back."""
    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        await websocket.close(code=1008, reason="Invalid session id")
        return
    if _pipeline is None:
        await websocket.close(code=1011, reason="Pipeline not initialized")
        return

    await websocket.accept()
    state = _ensure_session(normalized_session_id)
    bridge_state = _sonic_bridge_state.setdefault(normalized_session_id, SonicBridgeState())
    existing_websocket = _sonic_websockets.get(normalized_session_id)
    if existing_websocket is not None and existing_websocket is not websocket:
        with contextlib.suppress(Exception):
            await existing_websocket.close(code=1012, reason="Superseded by new Sonic connection")
    _sonic_websockets[normalized_session_id] = websocket

    # One live Sonic bridge per session. Replace existing on reconnect.
    await _stop_sonic_session(normalized_session_id)

    sonic = NovaSonicWebSocketSession(
        region=_pipeline.config.aws_region,
        model_id=_pipeline.config.nova_sonic_model_id,
        system_prompt=_SONIC_RUNTIME_SYSTEM_PROMPT,
    )
    _sonic_sessions[normalized_session_id] = sonic

    try:
        await sonic.start()
        await websocket.send_json(
            {
                "type": "ready",
                "session_id": normalized_session_id,
                "output_sample_rate_hz": sonic.output_sample_rate_hz,
            }
        )

        # Seed Sonic with existing screen OCR so it starts with awareness
        snapshot = _pipeline.get_session_snapshot(normalized_session_id)
        if snapshot and snapshot.ocr_text.strip():
            try:
                await sonic.send_context_update(
                    _clip_text(snapshot.ocr_text, 4000)
                )
                LOGGER.info(
                    "Seeded Sonic with existing OCR session=%s ocr_len=%d",
                    normalized_session_id,
                    len(snapshot.ocr_text),
                )
            except Exception as exc:
                LOGGER.warning("Failed seeding Sonic with OCR: %s", exc)
        if state.overlay_mermaid.strip():
            await websocket.send_json(
                {
                    "type": "overlay_diagram",
                    "mermaid": state.overlay_mermaid,
                    "prompt": state.overlay_prompt,
                    "source": "session_state",
                    "updated_at": state.overlay_updated_at.isoformat() if state.overlay_updated_at else None,
                    "position": state.overlay_position,
                    "minimized": state.overlay_minimized,
                }
            )

        async def _upstream_loop() -> None:
            def _avg_abs_pcm16(audio_bytes: bytes) -> float:
                if len(audio_bytes) < 2:
                    return 0.0
                samples = memoryview(audio_bytes).cast("h")
                total = sum(abs(int(sample)) for sample in samples)
                return float(total) / float(len(samples))

            audio_chunk_count = 0
            while True:
                payload = await websocket.receive_json()
                msg_type = str(payload.get("type", "")).strip()
                if msg_type == "audio_chunk":
                    raw_audio = str(payload.get("audio_base64", "")).strip()
                    if not raw_audio:
                        continue
                    try:
                        audio_bytes = base64.b64decode(raw_audio, validate=True)
                    except Exception:
                        LOGGER.debug(
                            "Discarding invalid Sonic audio chunk for session=%s",
                            normalized_session_id,
                        )
                        continue
                    await sonic.send_audio_chunk(audio_bytes)
                    audio_chunk_count += 1
                    if audio_chunk_count == 1 or audio_chunk_count % 100 == 0:
                        avg_abs = _avg_abs_pcm16(audio_bytes)
                        LOGGER.info(
                            "Sonic audio chunks session=%s count=%d last_chunk_bytes=%d avg_abs=%.1f",
                            normalized_session_id,
                            audio_chunk_count,
                            len(audio_bytes),
                            avg_abs,
                        )
                    continue

                if msg_type == "audio_turn_end":
                    LOGGER.info("Sonic audio turn end received for session=%s", normalized_session_id)
                    await asyncio.sleep(0.15)
                    user_prompt = _collapse_transcript_chunks(bridge_state.current_user_transcript_parts)
                    if not user_prompt:
                        # Avoid reusing stale prompts from prior turns.
                        user_prompt = ""
                    if user_prompt:
                        bridge_state.user_turn_count += 1
                        await _persist_sonic_memory(
                            _pipeline,
                            normalized_session_id,
                            state.run_id,
                            user_prompt,
                            "sonic_user_turn",
                            bridge_state.user_turn_count,
                        )
                        await _handle_sonic_overlay_intent(
                            _pipeline,
                            state,
                            user_prompt,
                            websocket,
                        )
                    context_payload, retrieved_count = await _build_sonic_context_payload(
                        _pipeline,
                        normalized_session_id,
                        state.run_id,
                        user_prompt,
                        bridge_state,
                    )
                    context_source = "retrieval"
                    should_push_context = bool(context_payload)
                    if should_push_context:
                        await sonic.send_context_update(context_payload, interactive=True)
                        bridge_state.last_context_payload = context_payload
                        LOGGER.info(
                            "Applied Sonic context session=%s source=%s retrieved_hits=%d ocr_present=%s",
                            normalized_session_id,
                            context_source,
                            retrieved_count,
                            "[Current screen OCR]" in context_payload,
                        )
                        await websocket.send_json(
                            {
                                "type": "context_applied",
                                "retrieved_hits": retrieved_count,
                                "has_current_ocr": "[Current screen OCR]" in context_payload,
                                "source": context_source,
                            }
                        )

                    # Small guard to let context updates flush before ending user turn.
                    await asyncio.sleep(0.08)
                    bridge_state.current_user_transcript_parts.clear()
                    bridge_state.last_user_transcript = ""
                    await sonic.end_user_audio_turn()
                    continue

                if msg_type == "context_update":
                    context_text = str(payload.get("text", "")).strip()
                    if context_text:
                        await sonic.send_context_update(context_text[:4000])
                    continue

                if msg_type == "overlay_clear":
                    state.overlay_mermaid = ""
                    state.overlay_prompt = ""
                    state.overlay_updated_at = datetime.now(timezone.utc)
                    state.overlay_minimized = False
                    await websocket.send_json(
                        {
                            "type": "overlay_clear",
                            "source": "client_request",
                        }
                    )
                    continue

                if msg_type == "overlay_control":
                    action = str(payload.get("action", "")).strip().lower()
                    if action == "move":
                        position = str(payload.get("position", "")).strip().lower()
                        if position in {"top-left", "top-right", "bottom-left", "bottom-right", "center"}:
                            state.overlay_position = position
                            await websocket.send_json(
                                {
                                    "type": "overlay_control",
                                    "action": "move",
                                    "position": position,
                                    "source": "client_request",
                                }
                            )
                    elif action == "minimize":
                        state.overlay_minimized = True
                        await websocket.send_json(
                            {
                                "type": "overlay_control",
                                "action": "minimize",
                                "source": "client_request",
                            }
                        )
                    elif action == "expand":
                        state.overlay_minimized = False
                        await websocket.send_json(
                            {
                                "type": "overlay_control",
                                "action": "expand",
                                "source": "client_request",
                            }
                        )
                    elif action == "export" and state.overlay_mermaid.strip():
                        await websocket.send_json(
                            {
                                "type": "overlay_export",
                                "mermaid": state.overlay_mermaid,
                                "prompt": state.overlay_prompt,
                                "session_id": state.session_id,
                                "source": "client_request",
                            }
                        )
                    continue

                if msg_type == "stop":
                    return

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

        async def _downstream_loop() -> None:
            downstream_count = 0
            while True:
                event = await sonic.next_event()
                downstream_count += 1
                event_type = str(event.get("type", ""))
                if event_type == "transcript":
                    role = str(event.get("role", "")).lower()
                    text = str(event.get("text", ""))
                    if role == "user":
                        _append_transcript_chunk(bridge_state.current_user_transcript_parts, text)
                        bridge_state.last_user_transcript = _collapse_transcript_chunks(
                            bridge_state.current_user_transcript_parts
                        )
                    elif role == "assistant":
                        _append_transcript_chunk(bridge_state.current_assistant_transcript_parts, text)
                        state.sonic_assistant_speaking = True
                elif event_type == "content_end":
                    role = str(event.get("role", "")).lower()
                    if role == "assistant":
                        state.sonic_assistant_speaking = False
                        assistant_text = _collapse_transcript_chunks(bridge_state.current_assistant_transcript_parts)
                        if assistant_text:
                            bridge_state.assistant_turn_count += 1
                            await _persist_sonic_memory(
                                _pipeline,
                                normalized_session_id,
                                state.run_id,
                                assistant_text,
                                "sonic_assistant_turn",
                                bridge_state.assistant_turn_count,
                            )
                        bridge_state.current_assistant_transcript_parts.clear()

                        # Flush pending OCR now that assistant finished speaking (with cooldown)
                        now_mono = time.monotonic()
                        if (
                            state.pending_sonic_ocr
                            and now_mono - state.last_sonic_ocr_push_at >= 30.0
                        ):
                            try:
                                await sonic.send_context_update(state.pending_sonic_ocr)
                                LOGGER.info(
                                    "Flushed pending OCR to Sonic session=%s ocr_len=%d",
                                    normalized_session_id,
                                    len(state.pending_sonic_ocr),
                                )
                                state.pending_sonic_ocr = ""
                                state.last_sonic_ocr_push_at = now_mono
                            except Exception as exc:
                                LOGGER.warning("Failed flushing pending OCR to Sonic: %s", exc)

                    elif role == "user":
                        # Keep user transcript chunks until frontend emits audio_turn_end.
                        pass
                elif event_type == "assistant_interrupted":
                    state.sonic_assistant_speaking = False
                    bridge_state.current_assistant_transcript_parts.clear()
                if (
                    downstream_count == 1
                    or downstream_count % 50 == 0
                    or event_type in {"content_end", "error", "transcript", "assistant_interrupted", "session_end"}
                ):
                    LOGGER.info(
                        "Sonic downstream session=%s count=%d type=%s role=%s text_len=%d",
                        normalized_session_id,
                        downstream_count,
                        event_type,
                        str(event.get("role", "")),
                        len(str(event.get("text", ""))),
                    )
                await websocket.send_json(event)

        upstream_task = asyncio.create_task(_upstream_loop(), name=f"sonic-upstream-{normalized_session_id}")
        downstream_task = asyncio.create_task(_downstream_loop(), name=f"sonic-downstream-{normalized_session_id}")

        done, pending = await asyncio.wait(
            {upstream_task, downstream_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(asyncio.CancelledError):
                exc = task.exception()
                if exc is not None:
                    raise exc
    except WebSocketDisconnect:
        LOGGER.info("Sonic websocket disconnected for session=%s", normalized_session_id)
    except Exception as exc:
        LOGGER.exception("Sonic websocket error for session=%s", normalized_session_id)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        await _stop_sonic_session(normalized_session_id)
        _sonic_bridge_state.pop(normalized_session_id, None)
        if _sonic_websockets.get(normalized_session_id) is websocket:
            _sonic_websockets.pop(normalized_session_id, None)
        with contextlib.suppress(Exception):
            await websocket.close()


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve local frontend when present."""
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index)


@app.get("/eye.svg", include_in_schema=False)
async def eye_icon() -> FileResponse:
    """Serve frontend favicon when available."""
    icon = _FRONTEND_DIR / "eye.svg"
    if not icon.exists():
        raise HTTPException(status_code=404, detail="icon not found")
    return FileResponse(icon)



def main() -> None:
    """Run API server."""
    uvicorn.run(app, host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
