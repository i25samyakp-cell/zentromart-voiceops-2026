"""Gnani-only speech input and output for the capstone."""

from __future__ import annotations

from pathlib import Path

from .config import Settings


class GnaniVoice:
    def __init__(self, settings: Settings):
        if not settings.gnani_api_key:
            raise RuntimeError(
                "GNANI_API_KEY is missing. Copy .env.example to .env and add "
                "your personal Gnani key."
            )
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        # The SDK also reads GNANI_API_KEY from the environment. Passing it
        # explicitly keeps configuration predictable without logging it.
        self.stt = GnaniSTTClient(api_key=settings.gnani_api_key)
        self.tts = GnaniTTSClient(api_key=settings.gnani_api_key)

    def transcribe(self, audio_path: Path, language_code: str = "en-IN") -> str:
        result = self.stt.transcribe(str(audio_path), language_code=language_code)
        return str(result.get("transcript", "")).strip()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "en-IN",
        voice: str = "Kaveri",
        model: str = "timbre-v2.5",
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tts.synthesize(
            text,
            voice=voice,
            model=model,
            language=language,
            output_file=str(output_path),
        )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Gnani TTS did not create a non-empty output file.")
        return output_path

