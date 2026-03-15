"""Nova Sonic real-time voice streaming with barge-in support."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import uuid
from typing import Any

import boto3
import pyaudio

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.identity import (
    AWSCredentialsIdentity,
    AWSIdentityProperties,
)
from smithy_core.aio.interfaces.identity import IdentityResolver
from smithy_core.exceptions import SmithyIdentityError

LOGGER = logging.getLogger(__name__)


class _Boto3CredentialsResolver(IdentityResolver[AWSCredentialsIdentity, AWSIdentityProperties]):
    """Resolve AWS credentials via boto3's default provider chain."""

    def __init__(self) -> None:
        self._session = boto3.Session()

    async def get_identity(self, *, properties: AWSIdentityProperties) -> AWSCredentialsIdentity:
        del properties  # Unused; boto3 resolves using its own provider chain.
        credentials = self._session.get_credentials()
        if credentials is None:
            raise SmithyIdentityError(
                "Unable to resolve AWS credentials for Nova Sonic. "
                "Configure AWS credentials (env vars, shared credentials file, or role)."
            )
        frozen = credentials.get_frozen_credentials()
        if not frozen.access_key or not frozen.secret_key:
            raise SmithyIdentityError(
                "Resolved AWS credentials are incomplete for Nova Sonic streaming."
            )
        return AWSCredentialsIdentity(
            access_key_id=frozen.access_key,
            secret_access_key=frozen.secret_key,
            session_token=frozen.token,
        )


def _create_bedrock_runtime_client(region: str) -> BedrockRuntimeClient:
    """Create a Bedrock Runtime streaming client with boto3-backed credentials."""
    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=_Boto3CredentialsResolver(),
    )
    return BedrockRuntimeClient(config=config)


def _extract_stream_error_message(payload: Any) -> str | None:
    """Extract a readable message from non-chunk stream variants."""
    payload_type = type(payload).__name__
    payload_value = getattr(payload, "value", None)
    if payload_value is None:
        return None
    if isinstance(getattr(payload_value, "bytes_", None), (bytes, bytearray)):
        return None

    message = getattr(payload_value, "message", None)
    if isinstance(message, str) and message.strip():
        return f"{payload_type}: {message.strip()}"
    return f"{payload_type}: {payload_value}"


class NovaSonicConsoleSession:
    """Console speech-to-speech session for Nova Sonic with barge-in support."""

    def __init__(
        self,
        region: str,
        model_id: str,
        system_prompt: str,
        input_sample_rate_hz: int = 16000,
        output_sample_rate_hz: int = 24000,
        channels: int = 1,
        chunk_size: int = 512,
    ) -> None:
        """Initialize stream session and audio settings."""
        self._region = region
        self._model_id = model_id
        self._system_prompt = system_prompt

        self._input_sample_rate_hz = input_sample_rate_hz
        self._output_sample_rate_hz = output_sample_rate_hz
        self._channels = channels
        self._chunk_size = chunk_size

        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._barge_in_triggered = False

        self._audio_input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self._bedrock_client = self._create_bedrock_client()
        self._stream_response: Any = None

        self._prompt_name = str(uuid.uuid4())
        self._system_content_name = str(uuid.uuid4())
        self._audio_content_name = str(uuid.uuid4())
        self._audio_content_has_data = False

        self._audio = pyaudio.PyAudio()
        self._input_stream: pyaudio.Stream | None = None
        self._output_stream: pyaudio.Stream | None = None

        self._receive_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._play_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        """Start streaming session and run until user stops it."""
        self._loop = asyncio.get_running_loop()
        await self._open_stream()
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop(), name="nova-sonic-receive")
        await self._send_initial_events()
        self._start_audio_streams()

        self._send_task = asyncio.create_task(self._send_audio_loop(), name="nova-sonic-send-audio")
        self._play_task = asyncio.create_task(self._play_audio_loop(), name="nova-sonic-play")

        LOGGER.info("Nova Sonic session started. Speak now; press Enter to stop.")
        try:
            await asyncio.get_running_loop().run_in_executor(None, input)
        finally:
            await self.close()

    async def close(self) -> None:
        """Close streaming resources and audio devices."""
        if not self._running and self._stream_response is None:
            return

        self._running = False

        with contextlib.suppress(Exception):
            await self._send_event(
                {
                    "event": {
                        "contentEnd": {
                            "promptName": self._prompt_name,
                            "contentName": self._audio_content_name,
                        }
                    }
                }
            )
        with contextlib.suppress(Exception):
            await self._send_event({"event": {"promptEnd": {"promptName": self._prompt_name}}})
        with contextlib.suppress(Exception):
            await self._send_event({"event": {"sessionEnd": {}}})

        await self._cancel_task(self._receive_task)
        await self._cancel_task(self._send_task)
        await self._cancel_task(self._play_task)

        if self._stream_response is not None:
            with contextlib.suppress(Exception):
                await self._stream_response.input_stream.close()
            self._stream_response = None

        if self._input_stream is not None:
            if self._input_stream.is_active():
                self._input_stream.stop_stream()
            self._input_stream.close()
            self._input_stream = None

        if self._output_stream is not None:
            if self._output_stream.is_active():
                self._output_stream.stop_stream()
            self._output_stream.close()
            self._output_stream = None

        self._audio.terminate()
        LOGGER.info("Nova Sonic session closed")

    def _create_bedrock_client(self) -> BedrockRuntimeClient:
        """Build Bedrock Runtime SDK client for bidirectional streaming."""
        return _create_bedrock_runtime_client(self._region)

    async def _open_stream(self) -> None:
        """Open bidirectional model stream."""
        self._stream_response = await self._bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model_id)
        )

    async def _send_initial_events(self) -> None:
        """Send session, prompt, and initial system text events."""
        await self._send_event(
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": 1024,
                            "topP": 0.9,
                            "temperature": 0.7,
                        },
                        "turnDetectionConfiguration": {
                            "endpointingSensitivity": "MEDIUM",
                        },
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "promptStart": {
                        "promptName": self._prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": self._output_sample_rate_hz,
                            "sampleSizeBits": 16,
                            "channelCount": self._channels,
                            "voiceId": "matthew",
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content_name,
                        "type": "TEXT",
                        "role": "SYSTEM",
                        "interactive": False,
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content_name,
                        "content": self._system_prompt,
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content_name,
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": self._input_sample_rate_hz,
                            "sampleSizeBits": 16,
                            "channelCount": self._channels,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )

    def _start_audio_streams(self) -> None:
        """Open local microphone input and speaker output streams."""
        self._input_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._input_sample_rate_hz,
            input=True,
            frames_per_buffer=self._chunk_size,
            stream_callback=self._input_callback,
        )
        self._output_stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._output_sample_rate_hz,
            output=True,
            frames_per_buffer=self._chunk_size,
        )

    def _input_callback(self, in_data, frame_count, time_info, status):  # noqa: ANN001, D401
        """Collect microphone chunks and forward to async queue."""
        if self._running and in_data and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._audio_input_queue.put(in_data), self._loop)
        return (None, pyaudio.paContinue)

    async def _send_audio_loop(self) -> None:
        """Forward captured microphone chunks to the Sonic stream."""
        while self._running:
            audio_bytes = await self._audio_input_queue.get()
            encoded = base64.b64encode(audio_bytes).decode("utf-8")
            await self._send_event(
                {
                    "event": {
                        "audioInput": {
                            "promptName": self._prompt_name,
                            "contentName": self._audio_content_name,
                            "content": encoded,
                        }
                    }
                }
            )

    async def _receive_loop(self) -> None:
        """Receive bidirectional events from Nova Sonic."""
        output_stream: Any
        try:
            _, output_stream = await self._stream_response.await_output()
        except Exception:
            LOGGER.exception("Failed to initialize Nova Sonic output stream")
            self._running = False
            return

        while self._running:
            try:
                payload = await output_stream.receive()
            except StopAsyncIteration:
                LOGGER.info("Nova Sonic stream ended")
                self._running = False
                break
            except Exception:
                LOGGER.exception("Failed receiving Nova Sonic stream event")
                self._running = False
                break

            payload_value = getattr(payload, "value", None)
            payload_bytes = getattr(payload_value, "bytes_", None)
            if not isinstance(payload_bytes, (bytes, bytearray)):
                error_message = _extract_stream_error_message(payload)
                if error_message:
                    LOGGER.error("Nova Sonic stream error: %s", error_message)
                    self._running = False
                    break
                continue

            try:
                data = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError:
                LOGGER.debug("Ignoring non-JSON Sonic payload")
                continue

            event = data.get("event", {})

            if "textOutput" in event:
                text_output = event["textOutput"]
                text_content = text_output.get("content", "")
                role = text_output.get("role", "")
                if role == "USER" and text_content:
                    LOGGER.info("User: %s", text_content)
                elif role == "ASSISTANT" and text_content:
                    LOGGER.info("Assistant: %s", text_content)

                if "\"interrupted\"" in text_content:
                    self._barge_in_triggered = True

            if "contentEnd" in event:
                stop_reason = event["contentEnd"].get("stopReason", "")
                if stop_reason == "INTERRUPTED":
                    self._barge_in_triggered = True

            if "audioOutput" in event:
                audio_content = event["audioOutput"].get("content", "")
                if audio_content:
                    await self._audio_output_queue.put(base64.b64decode(audio_content))

    async def _play_audio_loop(self) -> None:
        """Play assistant audio and clear buffered speech on barge-in."""
        while self._running:
            if self._barge_in_triggered:
                while not self._audio_output_queue.empty():
                    self._audio_output_queue.get_nowait()
                self._barge_in_triggered = False
                await asyncio.sleep(0.03)
                continue

            audio_data = await self._audio_output_queue.get()
            if self._output_stream is None:
                continue
            await asyncio.get_running_loop().run_in_executor(None, self._output_stream.write, audio_data)

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Serialize and send one event chunk to Bedrock input stream."""
        if self._stream_response is None:
            raise RuntimeError("Nova Sonic stream is not initialized")

        payload = json.dumps(event).encode("utf-8")
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload)
        )
        await self._stream_response.input_stream.send(chunk)

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        """Cancel and await an asyncio task if it exists."""
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class NovaSonicWebSocketSession:
    """Bidirectional Nova Sonic session for browser WebSocket bridges."""

    def __init__(
        self,
        region: str,
        model_id: str,
        system_prompt: str,
        input_sample_rate_hz: int = 16000,
        output_sample_rate_hz: int = 24000,
        channels: int = 1,
    ) -> None:
        """Initialize stream session and audio format metadata."""
        self._region = region
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._input_sample_rate_hz = input_sample_rate_hz
        self._output_sample_rate_hz = output_sample_rate_hz
        self._channels = channels

        self._running = False
        self._bedrock_client = self._create_bedrock_client()
        self._stream_response: Any = None

        self._prompt_name = str(uuid.uuid4())
        self._system_content_name = str(uuid.uuid4())
        self._audio_content_name = str(uuid.uuid4())
        self._audio_content_has_data = False

        self._receive_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._current_text_role = ""

    async def start(self) -> None:
        """Open Sonic stream and send initial session events."""
        if self._running:
            return
        await self._open_stream()
        self._running = True
        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name=f"nova-sonic-ws-receive-{self._prompt_name}",
        )
        try:
            await self._send_initial_events()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Close Sonic stream and background tasks."""
        if not self._running and self._stream_response is None:
            return
        self._running = False

        if self._audio_content_has_data:
            with contextlib.suppress(Exception):
                await self._send_event(
                    {
                        "event": {
                            "contentEnd": {
                                "promptName": self._prompt_name,
                                "contentName": self._audio_content_name,
                            }
                        }
                    }
                )
        with contextlib.suppress(Exception):
            await self._send_event({"event": {"promptEnd": {"promptName": self._prompt_name}}})
        with contextlib.suppress(Exception):
            await self._send_event({"event": {"sessionEnd": {}}})

        await self._cancel_task(self._receive_task)
        self._receive_task = None

        if self._stream_response is not None:
            with contextlib.suppress(Exception):
                await self._stream_response.input_stream.close()
            self._stream_response = None

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        """Forward one PCM16 chunk from browser microphone to Sonic."""
        if not self._running:
            return
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        await self._send_event(
            {
                "event": {
                    "audioInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "content": encoded,
                    }
                }
            }
        )
        self._audio_content_has_data = True

    async def end_user_audio_turn(self) -> None:
        """Close current audio content turn and start a fresh one for next utterance."""
        if not self._running or not self._audio_content_has_data:
            return
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                    }
                }
            }
        )
        self._audio_content_name = str(uuid.uuid4())
        self._audio_content_has_data = False
        await self._send_audio_content_start_event()

    async def send_context_update(self, context_text: str) -> None:
        """Inject latest visual context so Sonic stays aligned with screen state."""
        if not self._running:
            return
        normalized = context_text.strip()
        if not normalized:
            return

        context_content_name = str(uuid.uuid4())
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": context_content_name,
                        "type": "TEXT",
                        "role": "USER",
                        "interactive": False,
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": context_content_name,
                        "content": f"[Visual context update]\n{normalized}",
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": context_content_name,
                    }
                }
            }
        )

    async def next_event(self) -> dict[str, Any]:
        """Fetch next event emitted from Nova Sonic stream."""
        return await self._events.get()

    @property
    def output_sample_rate_hz(self) -> int:
        """Assistant output sample rate emitted by Sonic."""
        return self._output_sample_rate_hz

    def _create_bedrock_client(self) -> BedrockRuntimeClient:
        """Build Bedrock Runtime SDK client for bidirectional streaming."""
        return _create_bedrock_runtime_client(self._region)

    async def _open_stream(self) -> None:
        """Open bidirectional model stream."""
        self._stream_response = await self._bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model_id)
        )

    async def _send_initial_events(self) -> None:
        """Send session, prompt, and system/audio content start events."""
        await self._send_event(
            {
                "event": {
                    "sessionStart": {
                        "inferenceConfiguration": {
                            "maxTokens": 1024,
                            "topP": 0.9,
                            "temperature": 0.5,
                        },
                        "turnDetectionConfiguration": {
                            "endpointingSensitivity": "MEDIUM",
                        },
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "promptStart": {
                        "promptName": self._prompt_name,
                        "textOutputConfiguration": {"mediaType": "text/plain"},
                        "audioOutputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": self._output_sample_rate_hz,
                            "sampleSizeBits": 16,
                            "channelCount": self._channels,
                            "voiceId": "matthew",
                            "encoding": "base64",
                            "audioType": "SPEECH",
                        },
                    }
                }
            }
        )

        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content_name,
                        "type": "TEXT",
                        "role": "SYSTEM",
                        "interactive": False,
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "textInput": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content_name,
                        "content": self._system_prompt,
                    }
                }
            }
        )
        await self._send_event(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._prompt_name,
                        "contentName": self._system_content_name,
                    }
                }
            }
        )

        self._audio_content_has_data = False
        await self._send_audio_content_start_event()

    async def _send_audio_content_start_event(self) -> None:
        """Start a new USER audio input content block."""
        await self._send_event(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._prompt_name,
                        "contentName": self._audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": self._input_sample_rate_hz,
                            "sampleSizeBits": 16,
                            "channelCount": self._channels,
                            "audioType": "SPEECH",
                            "encoding": "base64",
                        },
                    }
                }
            }
        )

    async def _receive_loop(self) -> None:
        """Read Sonic stream events and push normalized payloads for websocket forwarding."""
        output_stream: Any
        try:
            _, output_stream = await self._stream_response.await_output()
        except Exception as exc:
            LOGGER.exception("Failed to initialize Nova Sonic output stream")
            await self._events.put({"type": "error", "message": str(exc)})
            self._running = False
            return

        while self._running:
            try:
                payload = await output_stream.receive()
                LOGGER.debug("Received Sonic stream payload: %s", type(payload).__name__)
            except StopAsyncIteration:
                await self._events.put({"type": "session_end", "reason": "stream_closed"})
                self._running = False
                break
            except Exception as exc:
                LOGGER.exception("Failed receiving Nova Sonic stream event")
                await self._events.put({"type": "error", "message": str(exc)})
                self._running = False
                break

            payload_value = getattr(payload, "value", None)
            payload_bytes = getattr(payload_value, "bytes_", None)
            if not isinstance(payload_bytes, (bytes, bytearray)):
                error_message = _extract_stream_error_message(payload)
                if error_message:
                    LOGGER.error("Nova Sonic stream error: %s", error_message)
                    await self._events.put({"type": "error", "message": error_message})
                    self._running = False
                    break
                continue

            try:
                data = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError:
                LOGGER.debug("Ignoring non-JSON Sonic payload")
                continue

            event = data.get("event", {})
            if not isinstance(event, dict):
                continue

            content_start = event.get("contentStart")
            if isinstance(content_start, dict):
                role = content_start.get("role")
                if isinstance(role, str):
                    self._current_text_role = role.lower()

            text_output = event.get("textOutput")
            if isinstance(text_output, dict):
                role = str(text_output.get("role", "")).lower() or self._current_text_role
                content = str(text_output.get("content", "")).strip()
                if content:
                    await self._events.put({"type": "transcript", "role": role, "text": content})

            audio_output = event.get("audioOutput")
            if isinstance(audio_output, dict):
                content = audio_output.get("content")
                if isinstance(content, str) and content:
                    await self._events.put(
                        {
                            "type": "assistant_audio",
                            "audio_base64": content,
                            "sample_rate_hz": self._output_sample_rate_hz,
                        }
                    )

            content_end = event.get("contentEnd")
            if isinstance(content_end, dict) and content_end.get("stopReason") == "INTERRUPTED":
                await self._events.put({"type": "assistant_interrupted"})

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Serialize and send one event chunk to Bedrock input stream."""
        if self._stream_response is None:
            raise RuntimeError("Nova Sonic stream is not initialized")

        payload = json.dumps(event).encode("utf-8")
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=payload)
        )
        async with self._send_lock:
            await self._stream_response.input_stream.send(chunk)

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        """Cancel and await an asyncio task if it exists."""
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


DEFAULT_SONIC_SYSTEM_PROMPT = (
    "You are a concise, real-time learning assistant. "
    "Answer clearly in one to three short sentences unless the user asks for depth."
)
