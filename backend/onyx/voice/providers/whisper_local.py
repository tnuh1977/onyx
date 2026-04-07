"""Local Whisper voice provider for STT.

Connects to a self-hosted OpenAI-compatible Whisper server (e.g. faster-whisper-server,
whisper.cpp HTTP server, LocalAI, Ollama) via a user-supplied base URL.

Only Speech-to-Text is supported; Text-to-Speech is not available.
Streaming transcription via the OpenAI Realtime API is not supported because most
local implementations only expose the batch ``/v1/audio/transcriptions`` endpoint.
"""

import io
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from onyx.voice.interface import VoiceProviderInterface

if TYPE_CHECKING:
    from openai import AsyncOpenAI

LOCAL_WHISPER_STT_MODELS = [{"id": "whisper-1", "name": "Whisper v1"}]


class LocalWhisperVoiceProvider(VoiceProviderInterface):
    """Voice provider for a locally-hosted, OpenAI-compatible Whisper server.

    The server must expose ``POST /v1/audio/transcriptions`` (same contract as
    the OpenAI Whisper REST API).  An API key is optional – many self-hosted
    servers do not require one.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str | None = None,
        stt_model: str | None = None,
        tts_model: str | None = None,
        default_voice: str | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        # Use a dummy key so the OpenAI client does not complain about a missing key.
        # Local servers that require authentication should store the real key.
        self.api_key = api_key or "local"
        self.stt_model = stt_model or "whisper-1"
        # TTS is not supported by local Whisper – ignore these fields.
        self._tts_model = tts_model
        self._default_voice = default_voice

        self._client: "AsyncOpenAI | None" = None

    def _get_client(self) -> "AsyncOpenAI":
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=f"{self.api_base}/v1",
            )
        return self._client

    async def transcribe(self, audio_data: bytes, audio_format: str) -> str:
        """Transcribe audio via the local Whisper server.

        Args:
            audio_data: Raw audio bytes.
            audio_format: File extension / MIME hint (e.g. ``"webm"``, ``"wav"``).

        Returns:
            Transcribed text string.
        """
        client = self._get_client()

        audio_file = io.BytesIO(audio_data)
        audio_file.name = f"audio.{audio_format}"

        response = await client.audio.transcriptions.create(
            model=self.stt_model,
            file=audio_file,
        )
        return response.text

    def synthesize_stream(
        self, text: str, voice: str | None = None, speed: float = 1.0
    ) -> AsyncIterator[bytes]:
        """Not supported – local Whisper provides STT only."""
        raise NotImplementedError(
            "Local Whisper does not support text-to-speech. "
            "Configure a dedicated TTS provider instead."
        )

    async def validate_credentials(self) -> None:
        """Verify that the local Whisper server is reachable.

        Attempts ``GET /v1/models``.  A successful HTTP response of any status
        code (including 401/404) confirms the server is reachable.  Only a
        connection-level failure (refused, timeout, DNS) is treated as an error,
        because many local implementations do not expose ``/v1/models`` and may
        return 404.

        A 5xx response is also treated as an error because it suggests the server
        is running but in an unhealthy state.
        """
        import aiohttp

        from onyx.utils.logger import setup_logger

        logger = setup_logger()
        url = f"{self.api_base}/v1/models"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status >= 500:
                        raise RuntimeError(
                            f"Local Whisper server returned server error {resp.status} "
                            f"when checking {url}."
                        )
                    logger.debug(
                        f"LocalWhisperVoiceProvider: health check {url} → {resp.status}"
                    )
        except aiohttp.ClientConnectorError as exc:
            raise RuntimeError(
                f"Cannot connect to the local Whisper server at '{self.api_base}'. "
                f"Check that the server is running and the URL is correct. ({exc})"
            ) from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(
                f"Failed to reach the local Whisper server at '{self.api_base}': {exc}"
            ) from exc

    def get_available_voices(self) -> list[dict[str, str]]:
        """Local Whisper has no TTS voices."""
        return []

    def get_available_stt_models(self) -> list[dict[str, str]]:
        """Return the supported STT model list."""
        return LOCAL_WHISPER_STT_MODELS.copy()

    def get_available_tts_models(self) -> list[dict[str, str]]:
        """Local Whisper has no TTS models."""
        return []

    def supports_streaming_stt(self) -> bool:
        """Local Whisper does not support the OpenAI Realtime streaming API."""
        return False

    def supports_streaming_tts(self) -> bool:
        return False
