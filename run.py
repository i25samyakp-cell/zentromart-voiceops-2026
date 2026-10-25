#!/usr/bin/env python3
"""Required one-command batch runner for customer audio files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agent import ZentroMartAgent
from src.config import get_settings
from src.utils import safe_filename, write_json
from src.voice import GnaniVoice


AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe customer audio, answer with the grounded agent, and synthesize replies."
    )
    parser.add_argument("--input-audio", required=True, type=Path, help="Folder containing audio files")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="Output folder")
    parser.add_argument("--dataset", type=Path, help="Extracted dataset folder (overrides DATASET_DIR)")
    parser.add_argument("--language", default="en-IN", help="Gnani STT language code")
    parser.add_argument("--reply-language", default="en-IN", help="Gnani TTS language code")
    parser.add_argument("--voice", default="Kaveri", help="Gnani voice name")
    parser.add_argument("--tts-model", default="timbre-v2.5", help="Gnani TTS model")
    parser.add_argument(
        "--shared-session",
        action="store_true",
        help="Treat sorted files as successive turns of one conversation",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Skip embedding calls and use the local lexical retrieval fallback",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings(args.dataset)
    settings.validate_dataset()
    if not args.input_audio.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_audio}")

    audio_files = sorted(
        path for path in args.input_audio.iterdir() if path.suffix.casefold() in AUDIO_SUFFIXES
    )
    if not audio_files:
        raise ValueError(f"No supported audio files found in {args.input_audio}")

    voice = GnaniVoice(settings)
    agent = ZentroMartAgent(settings, use_embeddings=not args.lexical_only)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for number, audio_path in enumerate(audio_files, start=1):
        if not args.shared_session:
            agent.reset()
        item_dir = args.output / f"{number:03d}_{safe_filename(audio_path.stem)}"
        item_dir.mkdir(parents=True, exist_ok=True)

        transcript = voice.transcribe(audio_path, args.language)
        if transcript:
            result = agent.respond(transcript)
        else:
            result = None

        if result is None:
            reply = "Sorry, I didn't catch that. Could you say it again?"
            decision = "clarify"
            sources: list[dict] = []
            tools: list[dict] = []
        else:
            reply = result.reply
            decision = result.decision
            sources = result.sources
            tools = result.tool_calls

        verification_turn = any(
            call.get("name") == "lookup_order"
            and call.get("arguments", {}).get("verification_value") == "[REDACTED]"
            for call in tools
        )
        saved_transcript = (
            "[VERIFICATION RESPONSE REDACTED]" if verification_turn else transcript
        )

        reply_audio = item_dir / "response.wav"
        voice.synthesize(
            reply,
            reply_audio,
            language=args.reply_language,
            voice=args.voice,
            model=args.tts_model,
        )
        (item_dir / "transcript.txt").write_text(saved_transcript + "\n", encoding="utf-8")
        (item_dir / "response.txt").write_text(reply + "\n", encoding="utf-8")
        write_json(item_dir / "retrieved_sources.json", sources)
        write_json(item_dir / "tool_trace.json", tools)
        write_json(
            item_dir / "result.json",
            {
                "input_file": audio_path.name,
                "transcript": saved_transcript,
                "decision": decision,
                "response": reply,
                "response_audio": reply_audio.name,
                "source_ids": [source.get("source_id") for source in sources],
            },
        )
        manifest.append(
            {
                "input_file": audio_path.name,
                "output_directory": item_dir.name,
                "decision": decision,
                "success": True,
            }
        )
        print(f"[{number}/{len(audio_files)}] {audio_path.name} -> {item_dir}")

    write_json(args.output / "manifest.json", manifest)
    print(f"Completed {len(manifest)} audio file(s). Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

