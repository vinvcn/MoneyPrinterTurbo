"""
Segment-first 字幕生成。

segment-first 流水线里旁白按 segment 拼接，每段的音频偏移量在 TTS
阶段就已经精确到毫秒（见 ``segment_audio``）。这里直接把每段文本写到
自己的时间窗上，不再依赖 Whisper 转写或 TTS cue 聚合，因此字幕天然
与旁白对齐，也不会出现 00:00:00 占位行。
"""

from pathlib import Path

from loguru import logger

from app.services.segment_audio import ms_to_srt_timestamp


def build_segment_subtitles(segments: list[dict], subtitle_file: str) -> str:
    """
    按每段音频的真实偏移量生成 SRT。

    Args:
        segments: 至少包含 index/text/start_ms/duration_ms 的 segment 记录
            （来自 segment_audio.prepare_segment_audio().segments）。
        subtitle_file: 输出 SRT 路径。

    Returns:
        成功时返回 subtitle_file；没有可写字幕行时返回空字符串。
    """
    lines: list[str] = []
    subtitle_index = 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        duration_ms = int(segment.get("duration_ms") or 0)
        if not text or duration_ms <= 0:
            # 静音占位段没有可读文本，跳过而不是写零宽字幕。
            continue

        start_ms = int(segment.get("start_ms") or 0)
        subtitle_index += 1
        lines.append(
            f"{subtitle_index}\n"
            f"{ms_to_srt_timestamp(start_ms)} --> "
            f"{ms_to_srt_timestamp(start_ms + duration_ms)}\n"
            f"{text}\n"
        )

    if not lines:
        logger.warning("no subtitle lines generated from segments")
        return ""

    output_path = Path(subtitle_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        f"segment subtitles created: {subtitle_file}, lines={subtitle_index}"
    )
    return subtitle_file
