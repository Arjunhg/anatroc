"""Runtime configuration for Anatroc."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

LOGGER = logging.getLogger(__name__)

MODE_CAMERA = "camera"
MODE_SCREEN_SHARE = "screen_share"
MODE_AUTO = "auto"
VALID_MODES = {MODE_CAMERA, MODE_SCREEN_SHARE, MODE_AUTO}
VALID_NOVA_EMBED_DIMENSIONS = {256, 384, 1024, 3072}
VALID_NOVA_TEXT_TRUNCATION_MODES = {"START", "END", "NONE"}


def _required_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()

def _normalize_converse_model_id(model_id: str, env_name: str) -> str:
    """Normalize Nova conversational model IDs to inference-profile IDs."""
    if model_id.startswith("amazon.nova-"):
        normalized = f"us.{model_id}"
        LOGGER.warning(
            "%s=%s may not support on-demand Converse/streaming. "
            "Using inference profile ID: %s",
            env_name,
            model_id,
            normalized,
        )
        return normalized
    return model_id


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Invalid integer for %s: %s. Using default %d.", name, raw_value, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        LOGGER.warning("Invalid float for %s: %s. Using default %.2f.", name, raw_value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a safe fallback."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AssistantConfig:
    """Configuration for Nova models, storage, and runtime behavior."""

    assistant_mode: str
    log_level: str

    aws_region: str
    nova_lite_model_id: str
    nova_sonic_model_id: str
    nova_embed_model_id: str
    nova_embedding_dimension: int
    nova_text_truncation_mode: Literal["START", "END", "NONE"]

    database_url: str
    redis_url: str
    redis_ttl_seconds: int

    session_timeout_seconds: int
    screen_capture_interval_seconds: float
    max_retrieval_results: int
    sonic_context_max_results: int
    sonic_context_max_chars_per_hit: int
    sonic_context_max_total_chars: int
    sonic_context_retrieval_cooldown_seconds: float

    enable_aurora_writes: bool
    enable_redis_cache: bool
    enable_sonic_context_retrieval: bool

    @property
    def is_screen_share_mode(self) -> bool:
        """Return True when running in screen-share mode."""
        return self.assistant_mode == MODE_SCREEN_SHARE

    @property
    def is_auto_mode(self) -> bool:
        """Return True when mode auto-detection is enabled."""
        return self.assistant_mode == MODE_AUTO

    @classmethod
    def from_env(cls, mode_override: str | None = None) -> "AssistantConfig":
        """Build configuration from environment variables."""
        raw_mode = (mode_override or os.getenv("VISION_ASSISTANT_MODE", MODE_AUTO)).strip().lower()
        if raw_mode not in VALID_MODES:
            LOGGER.warning("Unknown assistant mode '%s'. Falling back to '%s'.", raw_mode, MODE_AUTO)
            raw_mode = MODE_AUTO

        aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or ""
        if not aws_region.strip():
            raise ValueError("Missing required environment variable: AWS_REGION")

        embedding_dimension = _env_int("NOVA_EMBEDDING_DIMENSION", 1024)
        if embedding_dimension not in VALID_NOVA_EMBED_DIMENSIONS:
            LOGGER.warning(
                "Invalid NOVA_EMBEDDING_DIMENSION=%s. Using default 1024.",
                embedding_dimension,
            )
            embedding_dimension = 1024

        text_truncation_mode = os.getenv("NOVA_TEXT_TRUNCATION_MODE", "END").strip().upper()
        if text_truncation_mode not in VALID_NOVA_TEXT_TRUNCATION_MODES:
            LOGGER.warning(
                "Invalid NOVA_TEXT_TRUNCATION_MODE=%s. Using default END.",
                text_truncation_mode,
            )
            text_truncation_mode = "END"

        return cls(
            assistant_mode=raw_mode,
            log_level=os.getenv("VISION_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).strip().upper(),
            aws_region=aws_region.strip(),
            nova_lite_model_id=_normalize_converse_model_id(
                _required_env("NOVA_LITE_MODEL_ID"),
                "NOVA_LITE_MODEL_ID",
            ),
            # Keep Sonic as configured; bidirectional streaming expects direct model IDs.
            nova_sonic_model_id=_required_env("NOVA_SONIC_MODEL_ID"),
            nova_embed_model_id=_required_env("NOVA_EMBED_MODEL_ID"),
            nova_embedding_dimension=embedding_dimension,
            nova_text_truncation_mode=text_truncation_mode,
            database_url=_required_env("DATABASE_URL"),
            redis_url=_required_env("REDIS_URL"),
            redis_ttl_seconds=_env_int("REDIS_TTL_SECONDS", 1800),
            session_timeout_seconds=_env_int("SESSION_TIMEOUT", 1800),
            screen_capture_interval_seconds=_env_float("SCREEN_CAPTURE_INTERVAL",4.0),
            max_retrieval_results=_env_int("MAX_RETRIEVAL_RESULTS", 5),
            sonic_context_max_results=_env_int("SONIC_CONTEXT_MAX_RESULTS", 4),
            sonic_context_max_chars_per_hit=_env_int("SONIC_CONTEXT_MAX_CHARS_PER_HIT", 350),
            sonic_context_max_total_chars=_env_int("SONIC_CONTEXT_MAX_TOTAL_CHARS", 1200),
            sonic_context_retrieval_cooldown_seconds=_env_float("SONIC_CONTEXT_RETRIEVAL_COOLDOWN_SECONDS", 5.0),
            enable_aurora_writes=_env_bool("ENABLE_AURORA_WRITES", True),
            enable_redis_cache=_env_bool("ENABLE_REDIS_CACHE", True),
            enable_sonic_context_retrieval=_env_bool("ENABLE_SONIC_CONTEXT_RETRIEVAL", True),
        )
