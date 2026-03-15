from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

from vision_learning_assistant import (
    AssistantConfig,
    MODE_AUTO,
    MODE_CAMERA,
    MODE_SCREEN_SHARE,
    VALID_MODES,
    VisionLearningPipeline,
)
from vision_learning_assistant.nova import DEFAULT_SONIC_SYSTEM_PROMPT, NovaSonicConsoleSession

LOGGER = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    """Configure application logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    """Add optional assistant mode selection to a parser."""
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=None,
        help=(
            f"Assistant mode: '{MODE_CAMERA}' for camera coaching, "
            f"'{MODE_SCREEN_SHARE}' for screen-share analysis, "
            f"'{MODE_AUTO}' for runtime auto-selection."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build minimal CLI parser for Nova runtime tasks."""
    parser = argparse.ArgumentParser(description="Vision Learning Assistant (Nova)")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run interactive Nova analysis mode")
    _add_mode_argument(run_parser)
    run_parser.add_argument("--session-id", default=None)
    run_parser.add_argument("--frame-path", default=None)
    run_parser.add_argument("--prompt", default=None)
    run_parser.add_argument("--call-id", default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--call-type", default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--no-demo", action="store_true", help=argparse.SUPPRESS)
    run_parser.add_argument("--video-track-override", default=None, help=argparse.SUPPRESS)

    analyze_parser = subparsers.add_parser("analyze-screen", help="Analyze a single screen frame")
    _add_mode_argument(analyze_parser)
    analyze_parser.add_argument("--frame-path", required=True)
    analyze_parser.add_argument("--prompt", required=True)
    analyze_parser.add_argument("--session-id", default=None)

    diagram_parser = subparsers.add_parser("diagram", help="Generate a Mermaid overlay diagram")
    _add_mode_argument(diagram_parser)
    diagram_parser.add_argument("--prompt", required=True)

    voice_parser = subparsers.add_parser("voice-chat", help="Run Nova Sonic speech-to-speech mode")
    _add_mode_argument(voice_parser)
    voice_parser.add_argument("--system-prompt", default=DEFAULT_SONIC_SYSTEM_PROMPT)

    return parser


async def run_interactive_session(pipeline: VisionLearningPipeline, session_id: str) -> None:
    """Run an interactive CLI loop for frame analysis."""
    LOGGER.info("Interactive session started: %s", session_id)
    LOGGER.info("Type 'exit' to stop.")

    while True:
        prompt = input("prompt> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            LOGGER.info("Interactive session ended.")
            return
        if not prompt:
            continue

        frame_path = input("frame_path> ").strip()
        if frame_path.lower() in {"exit", "quit"}:
            LOGGER.info("Interactive session ended.")
            return
        if not frame_path:
            LOGGER.warning("Frame path is required for screen analysis in this mode.")
            continue

        if not Path(frame_path).exists():
            LOGGER.error("Frame path does not exist: %s", frame_path)
            continue

        result = await pipeline.analyze_screen_file(
            session_id=session_id,
            frame_path=frame_path,
            prompt=prompt,
        )
        LOGGER.info("Nova response: %s", result.response_text)
        LOGGER.info("Retrieved context hits: %d", result.context_hits)


async def run_command(args: argparse.Namespace) -> None:
    """Handle command execution with lifecycle-managed pipeline."""
    session_id = getattr(args, "session_id", None) or getattr(args, "call_id", None) or str(uuid.uuid4())
    config = AssistantConfig.from_env(mode_override=getattr(args, "mode", None))
    configure_logging(config.log_level)

    if args.command == "voice-chat":
        sonic_session = NovaSonicConsoleSession(
            region=config.aws_region,
            model_id=config.nova_sonic_model_id,
            system_prompt=args.system_prompt,
        )
        await sonic_session.run()
        return

    pipeline = VisionLearningPipeline(config)
    await pipeline.start()

    try:
        if args.command == "run":
            if args.frame_path and args.prompt:
                result = await pipeline.analyze_screen_file(
                    session_id=session_id,
                    frame_path=args.frame_path,
                    prompt=args.prompt,
                )
                LOGGER.info("Nova response: %s", result.response_text)
                LOGGER.info("Retrieved context hits: %d", result.context_hits)
                return

            if sys.stdin.isatty():
                await run_interactive_session(pipeline, session_id=session_id)
            else:
                LOGGER.info(
                    "Headless run mode active for session %s. "
                    "Sleeping until SESSION_TIMEOUT (%ss).",
                    session_id,
                    config.session_timeout_seconds,
                )
                await asyncio.sleep(config.session_timeout_seconds)
            return

        if args.command == "analyze-screen":
            result = await pipeline.analyze_screen_file(
                session_id=session_id,
                frame_path=args.frame_path,
                prompt=args.prompt,
            )
            LOGGER.info("Nova response: %s", result.response_text)
            LOGGER.info(
                "Stored record id: %s | Retrieved context hits: %d",
                result.stored_record_id,
                result.context_hits,
            )
            return

        if args.command == "diagram":
            diagram = await pipeline.generate_overlay_diagram(prompt=args.prompt)
            LOGGER.info("Generated Mermaid diagram:\n%s", diagram)
            return

        raise ValueError(f"Unknown command: {args.command}")
    finally:
        await pipeline.close()


def main() -> None:
    """CLI entrypoint."""
    load_dotenv()
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    asyncio.run(run_command(args))


if __name__ == "__main__":
    main()
