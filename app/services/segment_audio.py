"""
Per-segment TTS generation and frame-accurate narration assembly.

The segment-first pipeline needs one audio file per narration segment plus a
single merged narration track whose segment offsets are known to the
millisecond — subtitle timing and clip assembly both derive from it.

Why pydub (already a project dependency) instead of ffmpeg concat or MoviePy:
- mp3 concat (`-c copy`) reports container durations with ~35ms padding per
  file, which drifts across many segments;
- MoviePy re-encodes and adds similar per-clip padding;
- pydub decodes to raw frames, so `len(segment)` is exact in milliseconds and
  offsets stay aligned with real audio content. Exporting to a single MP3
  keeps the downstream `generate_video()` path unchanged.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger
from pydub import AudioSegment

from app.services import voice as voice_service
from app.utils import utils


@dataclass
class SegmentAudioResult:
    """Result of preparing per-segment narration audio."""

    segments: List[dict] = field(default_factory=list)
    # Merged narration file (MP3) covering all segments in order.
    audio_file: str = ""
    total_duration_ms: int = 0
    ok: bool = True
    failed_index: Optional[int] = None
    error: str = ""


def ms_to_srt_timestamp(duration_ms: int) -> str:
    """Convert milliseconds to an SRT timestamp (HH:MM:SS,mmm)."""
    hours = duration_ms // 3_600_000
    minutes = (duration_ms % 3_600_000) // 60_000
    seconds = (duration_ms % 60_000) // 1000
    millis = duration_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def prepare_segment_audio(
    segments: List[dict],
    task_id: str,
    tts: Callable[..., Optional[object]],
    voice_name: Optional[str] = None,
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    sample_audio_base64: Optional[str] = None,
) -> SegmentAudioResult:
    """
    Run TTS per segment and merge the outputs into one narration track.

    Args:
        segments: segment dicts with at least {"index", "text"}.
        task_id: task directory owner for segment audio files.
        tts: TTS callable; defaults to `voice.tts` and receives
            text/voice_name/voice_rate/voice_file/voice_volume plus
            sample_audio_base64. Returning None marks the segment failed.
        voice_name/rate/volume/sample_audio_base64: forwarded to `tts`
            when the default is used.

    Returns:
        SegmentAudioResult with per-segment start/duration offsets (ms),
        the merged audio path, and total duration. `ok=False` on the first
        segment whose TTS failed; the task layer converts that to failure.
    """
    task_dir = Path(utils.task_dir(task_id))
    result = SegmentAudioResult()

    merged = AudioSegment.empty()
    offset_ms = 0

    for position, segment in enumerate(segments):
        index = int(segment.get("index", position))
        text = str(segment.get("text") or "").strip()
        segment_audio_path = str(task_dir / f"audio-segment-{index}.mp3")

        if text:
            kwargs = {
                "text": text,
                "voice_file": segment_audio_path,
                "voice_rate": voice_rate,
                "voice_volume": voice_volume,
            }
            if tts is _default_tts:
                kwargs.update(
                    {
                        "voice_name": voice_service.parse_voice_name(
                            voice_name or ""
                        ),
                        "sample_audio_base64": sample_audio_base64,
                    }
                )
            sub_maker = tts(**kwargs)
            if sub_maker is None:
                logger.error(
                    f"segment TTS failed: task_id={task_id}, segment={index}"
                )
                result.ok = False
                result.failed_index = index
                result.error = f"segment {index} TTS failed"
                return result
        else:
            # Empty segment text: keep the timeline but add silence instead
            # of calling TTS with nothing.
            merged += AudioSegment.silent(duration=500)

        if not text:
            result.segments.append(
                {
                    "index": index,
                    "text": text,
                    "audio_file": "",
                    "start_ms": offset_ms,
                    "duration_ms": 500,
                }
            )
            offset_ms += 500
            continue

        try:
            segment_audio = AudioSegment.from_file(segment_audio_path)
        except Exception as exc:
            logger.error(
                "failed to decode segment audio: "
                f"task_id={task_id}, segment={index}, "
                f"error={type(exc).__name__}, detail={exc}"
            )
            result.ok = False
            result.failed_index = index
            result.error = f"segment {index} audio is not decodable"
            return result

        duration_ms = len(segment_audio)
        merged += segment_audio
        result.segments.append(
            {
                "index": index,
                "text": text,
                "audio_file": segment_audio_path,
                "start_ms": offset_ms,
                "duration_ms": duration_ms,
            }
        )
        offset_ms += duration_ms
        logger.debug(
            f"segment {index} audio ready: task_id={task_id}, "
            f"duration_ms={duration_ms}, start_ms={result.segments[-1]['start_ms']}"
        )

    if not result.segments:
        return result

    merged_path = task_dir / "audio.mp3"
    # 192k matches the final video audio bitrate; mono keeps TTS output shape.
    merged.export(str(merged_path), format="mp3", bitrate="192k")

    result.audio_file = str(merged_path)
    result.total_duration_ms = offset_ms
    logger.info(
        f"merged segment narration audio: task_id={task_id}, "
        f"segments={len(result.segments)}, duration_ms={offset_ms}"
    )
    return result


def _default_tts(**kwargs) -> Optional[object]:
    return voice_service.tts(**kwargs)
