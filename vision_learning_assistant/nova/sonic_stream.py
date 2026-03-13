"""Nova Sonic real-time voice streaming with barge-in support."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import uuid
from typing import Any

import pyaudio
from aws_sdk_bedrock_runtime.auth import HTTPAuthSchemeResolver
from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
from smithy_core.shapes import ShapeID

LOGGER = logging.getLogger(__name__)


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
        await self._send_initial_events()
        self._start_audio_streams()

        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop(), name="nova-sonic-receive")
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
        config = Config(
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={
                ShapeID("com.amazonaws.bedrockruntime#aws.auth#sigv4"): SigV4AuthScheme(
                    service="bedrock",
                ),
            },
            endpoint_uri=f"https://bedrock-runtime.{self._region}.amazonaws.com",
            region=self._region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        return BedrockRuntimeClient(config=config)

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
                        "toolUseOutputConfiguration": {"mediaType": "application/json"},
                        "toolConfiguration": {"tools": []},
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
        while self._running:
            try:
                output = await self._stream_response.await_output()
                payload = await output[1].receive()
            except StopAsyncIteration:
                LOGGER.info("Nova Sonic stream ended")
                self._running = False
                break
            except Exception:
                LOGGER.exception("Failed receiving Nova Sonic stream event")
                self._running = False
                break

            if payload.value is None or payload.value.bytes_ is None:
                continue

            try:
                data = json.loads(payload.value.bytes_.decode("utf-8"))
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


DEFAULT_SONIC_SYSTEM_PROMPT = (
    "You are a concise, real-time learning assistant. "
    "Answer clearly in one to three short sentences unless the user asks for depth."
)
