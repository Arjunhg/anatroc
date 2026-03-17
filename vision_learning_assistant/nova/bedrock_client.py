"""Amazon Nova Bedrock client helpers for reasoning and embeddings."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Sequence

import boto3

LOGGER = logging.getLogger(__name__)

NOVA_EMBED_SCHEMA_VERSION = "nova-multimodal-embed-v1"
_VALID_EMBED_DIMENSIONS = {256, 384, 1024, 3072}
_VALID_TRUNCATION_MODES = {"START", "END", "NONE"}

EmbeddingPurpose = Literal[
    "GENERIC_INDEX",
    "GENERIC_RETRIEVAL",
    "TEXT_RETRIEVAL",
    "IMAGE_RETRIEVAL",
    "VIDEO_RETRIEVAL",
    "DOCUMENT_RETRIEVAL",
    "AUDIO_RETRIEVAL",
    "CLASSIFICATION",
    "CLUSTERING",
]

_OCR_EXTRACTION_PROMPT = (
    "Extract visible text from this screenshot. "
    "Return plain text only, preserve line breaks when useful, and do not add commentary."
)

class NovaBedrockClient:
    """Thin wrapper around Bedrock Runtime calls for Nova models."""

    def __init__(
        self,
        region: str,
        lite_model_id: str,
        embed_model_id: str,
        embedding_dimension: int = 1024,
        text_truncation_mode: Literal["START", "END", "NONE"] = "END",
    ) -> None:
        """Initialize Bedrock runtime clients and target model IDs."""
        self._lite_model_id = lite_model_id
        self._embed_model_id = embed_model_id

        if embedding_dimension not in _VALID_EMBED_DIMENSIONS:
            raise ValueError(
                f"Invalid embedding dimension {embedding_dimension}. "
                f"Allowed values: {sorted(_VALID_EMBED_DIMENSIONS)}"
            )
        if text_truncation_mode not in _VALID_TRUNCATION_MODES:
            raise ValueError(
                f"Invalid truncation mode {text_truncation_mode}. "
                f"Allowed values: {sorted(_VALID_TRUNCATION_MODES)}"
            )

        self._embedding_dimension = embedding_dimension
        self._text_truncation_mode = text_truncation_mode
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def embed_text(
        self,
        text: str,
        purpose: EmbeddingPurpose = "GENERIC_INDEX",
    ) -> list[float]:
        """Generate a Nova embedding vector for the provided text."""
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Cannot embed empty text")

        payload = {
            "schemaVersion": NOVA_EMBED_SCHEMA_VERSION,
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": purpose,
                "embeddingDimension": self._embedding_dimension,
                "text": {
                    "truncationMode": self._text_truncation_mode,
                    "value": normalized_text,
                },
            },
        }
        response = self._client.invoke_model(
            modelId=self._embed_model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        body = json.loads(response["body"].read())

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError(
                "Nova embedding response did not include 'embeddings'. "
                f"Response keys: {list(body.keys())}"
            )

        first_embedding = embeddings[0]
        if not isinstance(first_embedding, dict):
            raise ValueError("Nova embedding response has invalid embeddings item format.")

        vector = first_embedding.get("embedding")
        if not isinstance(vector, list):
            raise ValueError("Nova embedding response did not contain a numeric embedding vector.")

        float_vector = [float(value) for value in vector]
        if len(float_vector) != self._embedding_dimension:
            raise ValueError(
                "Nova embedding dimension mismatch. "
                f"Expected {self._embedding_dimension}, got {len(float_vector)}."
            )
        return float_vector

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """Extract OCR-like text from a screenshot using Nova Lite."""
        if not image_bytes:
            return ""

        response = self._client.converse(
            modelId=self._lite_model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": _OCR_EXTRACTION_PROMPT},
                        {
                            "image": {
                                "format": self._detect_image_format(image_bytes),
                                "source": {"bytes": image_bytes},
                            }
                        },
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 1600, "temperature": 0.0, "topP": 0.1},
        )

        extracted = self._extract_text_response(response).strip()
        return extracted

    def analyze_screen(
        self,
        prompt: str,
        image_bytes: bytes | None,
        current_ocr_text: str | None = None,
        retrieved_context: Sequence[str] | None = None,
    ) -> str:
        """Run Nova Lite screen reasoning using text plus optional frame bytes."""
        context_blocks = [block.strip() for block in (retrieved_context or []) if block.strip()]

        user_content: list[dict[str, Any]] = []
        normalized_ocr = (current_ocr_text or "").strip()
        if normalized_ocr:
            user_content.append(
                {
                    "text": (
                        "Current OCR text from the latest shared screen frame:\n"
                        f"{normalized_ocr}"
                    )
                }
            )
        if context_blocks:
            joined_context = "\n\n".join(context_blocks)
            user_content.append({"text": f"Earlier context from this same session:\n{joined_context}"})

        user_content.append({"text": f"User request:\n{prompt.strip()}"})

        if image_bytes is not None:
            user_content.append(
                {
                    "image": {
                        "format": self._detect_image_format(image_bytes),
                        "source": {"bytes": image_bytes},
                    }
                }
            )

        response = self._client.converse(
            modelId=self._lite_model_id,
            system=[
                {
                    "text": (
                        "You are a precise learning assistant. "
                        "Treat the current image and the current OCR text as the primary source of truth. "
                        "Use earlier retrieved context only as a secondary hint from the same session. "
                        "If earlier context conflicts with the current screen, ignore the earlier context. "
                        "Describe exactly what is visible now, not what may have appeared earlier. "
                        "Return concise, actionable guidance in plain language. "
                        "If the current screen is unreadable or insufficient, say exactly what is missing."
                    )
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            inferenceConfig={"maxTokens": 700, "temperature": 0.2, "topP": 0.9},
        )

        return self._extract_text_response(response)

    def generate_mermaid_diagram(self, prompt: str, retrieved_context: Sequence[str] | None = None) -> str:
        """Generate Mermaid diagram text from user intent and retrieved context."""
        context_blocks = [block.strip() for block in (retrieved_context or []) if block.strip()]
        joined_context = "\n\n".join(context_blocks)

        user_prompt = (
            "Generate a Mermaid diagram that answers this request:\n"
            f"{prompt.strip()}\n\n"
            "Rules:\n"
            "1. Output only Mermaid code.\n"
            "2. Start with graph TD or flowchart TD.\n"
            "3. Keep labels short and clear."
        )
        if joined_context:
            user_prompt += f"\n\nAdditional context:\n{joined_context}"

        response = self._client.converse(
            modelId=self._lite_model_id,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 800, "temperature": 0.1, "topP": 0.8},
        )

        return self._extract_text_response(response)

    # {
    #     "output": {
    #         "message": {
    #             "content": [
    #                 {"text": "This diagram shows a RAG pipeline."}
    #             ]
    #         }
    #     }
    # }
    @staticmethod
    def _extract_text_response(response: dict[str, Any]) -> str:
        """Extract concatenated text blocks from a Bedrock converse response."""
        output = response.get("output", {})
        message = output.get("message", {})
        blocks = message.get("content", [])
        text_parts = [block.get("text", "") for block in blocks if isinstance(block, dict) and "text" in block]
        result = "\n".join(part for part in text_parts if part).strip()
        if not result:
            raise ValueError("Nova Lite response did not include text content.")
        return result

    @staticmethod
    def _detect_image_format(image_bytes: bytes) -> str:
        """Best-effort content-type detection for Bedrock image blocks."""
        if image_bytes.startswith(b"\x89PNG"):
            return "png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return "gif"
        if image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "webp"
        LOGGER.debug("Unknown image header, defaulting to jpeg")
        return "jpeg"
