"""
逐段 TTS 生成与毫秒级旁白合并。

segment-first 流水线需要每个旁白片段独立的音频文件，以及一条合并后的
旁白轨道——字幕时间轴和素材组装都依赖每段在合并轨道上的偏移量。

选择 pydub（项目既有依赖）而不是 ffmpeg concat 或 MoviePy 的原因：
- mp3 concat（``-c copy``）每个文件会引入约 35ms 的容器填充，片段一多
  偏移就漂移；
- MoviePy 会重编码并引入类似的单片段填充；
- pydub 解码到原始帧，``len(segment)`` 以毫秒为单位精确，偏移量与真实
  音频内容对齐。导出单个 MP3 保持下游 ``generate_video()`` 路径不变。

解码细节：pydub 的 ``from_file`` 需要 ffprobe 探测容器信息，而
Windows 便携包和部分 CI 环境只有 ffmpeg 没有 ffprobe。这里统一先把
片段音频转成 WAV 流（Frame-accurate、无容器填充），再用 pydub 的纯
Python WAV 解析读取，两条环境约束都不破坏。
"""

import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger
from pydub import AudioSegment

from app.services import voice as voice_service
from app.utils import utils


def _pydub_segment() -> AudioSegment:
    """
    返回已配置 FFmpeg 路径的 AudioSegment。

    pydub 解码依赖外部 FFmpeg。CI/Windows 环境 PATH 里往往没有 ffmpeg，
    项目已通过 imageio-ffmpeg 提供内置二进制；这里在每次解码前复用
    voice 服务的同一套路径解析，避免 segment 音频在 CI/便携包环境
    "ffmpeg not found" 失败。
    """
    voice_service._configure_pydub_ffmpeg(AudioSegment)
    return AudioSegment


def _decode_audio_frames(audio_path: str) -> AudioSegment:
    """
    用 ffmpeg 把任意音频解码为 WAV 帧流，交给 pydub 解析。

    不经过 ffprobe，也不重新引入容器填充；WAV 路径的 ``len()`` 与源
    音频帧数完全一致，保证逐段偏移量在所有环境都可复现。
    """
    ffmpeg_binary = utils.get_ffmpeg_binary()
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-v", "error",
        "-i", audio_path,
        "-f", "wav",
        "-acodec", "pcm_s16le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        detail = (result.stderr or b"").decode("utf-8", "ignore").strip()
        raise ValueError(f"ffmpeg decode failed: {detail[:200]}")
    return AudioSegment.from_wav(io.BytesIO(result.stdout))


@dataclass
class SegmentAudioResult:
    """逐段配音结果：包含每段偏移量与合并后的旁白文件路径。"""

    segments: List[dict] = field(default_factory=list)
    # 覆盖全部片段、按顺序合并后的旁白文件（MP3）。
    audio_file: str = ""
    total_duration_ms: int = 0
    ok: bool = True
    failed_index: Optional[int] = None
    error: str = ""


def ms_to_srt_timestamp(duration_ms: int) -> str:
    """把毫秒转换为 SRT 时间戳（HH:MM:SS,mmm）。"""
    hours = duration_ms // 3_600_000
    minutes = (duration_ms % 3_600_000) // 60_000
    seconds = (duration_ms % 60_000) // 1000
    millis = duration_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def prepare_segment_audio(
    segments: List[dict],
    task_id: str,
    voice_name: str = "",
    voice_rate: float = 1.0,
    voice_volume: float = 1.0,
    sample_audio_base64: Optional[str] = None,
    tts: Optional[Callable[..., Optional[object]]] = None,
) -> SegmentAudioResult:
    """
    逐段调用 TTS 并合并为一条旁白轨道。

    Args:
        segments: 至少包含 {"index", "text"} 的片段记录。
        task_id: 片段音频文件所属的任务目录。
        voice_name/rate/volume/sample_audio_base64: 转发给 ``voice.tts``。
        tts: 可注入的 TTS 函数；缺省使用 ``voice.tts``，返回 None 表示
            该片段配音失败。

    Returns:
        SegmentAudioResult，包含每段的 start/duration 偏移（毫秒）、合并
        音频路径与总时长。任一片段 TTS 失败时 ``ok=False``，由任务层转换
        为任务失败状态。
    """
    task_dir = Path(utils.task_dir(task_id))
    result = SegmentAudioResult()
    # 测试需要替换 TTS 时注入；生产路径始终走 voice.tts。
    tts_callable = tts if tts is not None else voice_service.tts

    merged = AudioSegment.empty()
    offset_ms = 0

    for position, segment in enumerate(segments):
        index = int(segment.get("index", position))
        text = str(segment.get("text") or "").strip()
        segment_audio_path = str(task_dir / f"audio-segment-{index}.mp3")

        if not text:
            # 空片段保留时间轴占位（500ms 静音），不向 TTS 传空文本。
            merged += AudioSegment.silent(duration=500)
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

        sub_maker = tts_callable(
            text=text,
            voice_name=voice_service.parse_voice_name(voice_name or ""),
            voice_rate=voice_rate,
            voice_file=segment_audio_path,
            voice_volume=voice_volume,
            sample_audio_base64=sample_audio_base64,
        )
        if sub_maker is None:
            logger.error(
                f"segment TTS failed: task_id={task_id}, segment={index}"
            )
            result.ok = False
            result.failed_index = index
            result.error = f"segment {index} TTS failed"
            return result

        try:
            segment_audio = _decode_audio_frames(segment_audio_path)
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
    # 192k 与成片音频码率一致；单声道保持 TTS 输出形态。导出使用项目解析的
    # FFmpeg，CI/便携包环境 PATH 里没有 ffmpeg 也能写出。
    merged_path_str = str(merged_path)
    voice_service._configure_pydub_ffmpeg(AudioSegment)
    merged.export(merged_path_str, format="mp3", bitrate="192k")

    result.audio_file = str(merged_path)
    result.total_duration_ms = offset_ms
    logger.info(
        f"merged segment narration audio: task_id={task_id}, "
        f"segments={len(result.segments)}, duration_ms={offset_ms}"
    )
    return result
