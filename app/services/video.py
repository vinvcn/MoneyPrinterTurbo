import itertools
import io
import math
import os
import random
import gc
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from typing import List
from loguru import logger
import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    VideoFileClip,
    afx,
)
from moviepy.tools import compute_position
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageDraw, ImageFont

from app.config import config
from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services.utils import video_effects
from app.utils import file_security, utils

class SubClippedVideoClip:
    def __init__(
        self,
        file_path,
        start_time=None,
        end_time=None,
        width=None,
        height=None,
        duration=None,
        source_file_path=None,
    ):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        self.source_file_path = source_file_path or file_path
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Docker 里的 ffmpeg/AAC 组合在默认配置下更容易出现音频质量波动，
# 这里显式抬高音频码率，避免成片阶段因为默认值过低而引入明显失真。
audio_bitrate = "192k"
fps = 30
# FFmpeg 按帧率拼接/转码时，最终时长可能比 MoviePy 读到的理论时长短几十毫秒。
# 这里给视频素材多留一个很小的安全余量，避免音频末尾因为帧舍入出现黑屏、
# 卡顿或最后一小段旁白没有画面的情况。
_VIDEO_DURATION_SAFETY_MARGIN = 0.1
_MIN_MATERIAL_DIMENSION = 480
# 消息类应用和部分编码器会把画面尺寸向下取整，例如 WhatsApp 会把 9:16 的
# 素材压成 478x850，比 480 少两个像素。直接按 480 硬卡会让这类素材全部被
# 丢弃，最终以 "no valid materials found" 整体失败。这里留一个很小的容差，
# 既能放行仅仅因为取整而略低于阈值的素材，也仍然能挡住真正的低清素材。
_MIN_DIMENSION_TOLERANCE = 10
_DEFAULT_VIDEO_CODEC = "libx264"
_SUPPORTED_VIDEO_CODECS = (
    "libx264",
    "h264_nvenc",
    "h264_amf",
    "h264_qsv",
    "h264_mf",
    "h264_videotoolbox",
)
_runtime_disabled_video_codecs = set()

# ---------------------------------------------------------------------------
# Renderer mode switches (ADR-0013)
#
# The final-video render and the combine phase each select exactly one
# implementation through config.toml. The selected path is the only path: there
# is no runtime fallback and no circuit breaker. Values are validated once at
# startup (the API controller resolves both switches at import time) and again
# per task, so an unknown value fails loudly instead of on the first frame.
#
# Defaults live here, version-controlled; ``config.toml`` (gitignored, per
# deployment) overrides them. Follow-up tickets add "ffmpeg_overlay" /
# "ffmpeg_filter" as new mode values together with their implementation.
# ---------------------------------------------------------------------------
FINAL_RENDER_MODE_MOVIEPY = "moviepy"
DEFAULT_FINAL_RENDER_MODE = FINAL_RENDER_MODE_MOVIEPY
_SUPPORTED_FINAL_RENDER_MODES = (FINAL_RENDER_MODE_MOVIEPY,)

COMBINE_RENDER_MODE_MOVIEPY = "moviepy"
COMBINE_RENDER_MODE_FFMPEG_FILTER = "ffmpeg_filter"
DEFAULT_COMBINE_RENDER_MODE = COMBINE_RENDER_MODE_MOVIEPY
_SUPPORTED_COMBINE_RENDER_MODES = (
    COMBINE_RENDER_MODE_MOVIEPY,
    COMBINE_RENDER_MODE_FFMPEG_FILTER,
)


def resolve_render_mode(
    configured_mode,
    *,
    switch_name: str,
    supported_modes: tuple[str, ...],
    default_mode: str,
) -> str:
    """
    解析单个渲染器开关，未配置时使用代码内置默认值。

    与 ``task_execution_mode`` 的“不安全组合静默降级”不同，渲染器开关必须
    显式且穷尽：未知取值直接抛错（启动时即失败），绝不回退到其它实现，
    否则会重现 F4 那种静默降级的问题（见 ADR-0013）。
    """
    mode = str(configured_mode or default_mode).strip().lower()
    if mode not in supported_modes:
        raise ValueError(
            f"unsupported {switch_name}: {configured_mode!r}; "
            f"supported values: {', '.join(supported_modes)}"
        )
    return mode


def resolve_final_render_mode(configured_mode=None) -> str:
    """解析最终成片渲染器开关（``final_render_mode``）。"""
    return resolve_render_mode(
        configured_mode,
        switch_name="final_render_mode",
        supported_modes=_SUPPORTED_FINAL_RENDER_MODES,
        default_mode=DEFAULT_FINAL_RENDER_MODE,
    )


def resolve_combine_render_mode(configured_mode=None) -> str:
    """解析素材拼接阶段渲染器开关（``combine_render_mode``）。"""
    return resolve_render_mode(
        configured_mode,
        switch_name="combine_render_mode",
        supported_modes=_SUPPORTED_COMBINE_RENDER_MODES,
        default_mode=DEFAULT_COMBINE_RENDER_MODE,
    )


def validate_render_modes(config_app) -> tuple[str, str]:
    """
    在进程启动时一次性校验两个渲染器开关。

    返回 ``(final_render_mode, combine_render_mode)``；任一取值非法时抛出
    ``ValueError``，让服务启动失败而不是把错误推迟到第一个任务。
    """
    return (
        resolve_final_render_mode(config_app.get("final_render_mode")),
        resolve_combine_render_mode(config_app.get("combine_render_mode")),
    )


def _dispatch_renderer(
    *,
    switch_name: str,
    configured_mode,
    resolve,
    implementations: dict,
    label: str,
    **kwargs,
):
    """
    解析开关、查表并执行唯一实现，两个渲染阶段共用同一分派契约。

    选择的结果就是唯一执行的路径，没有运行时回退。所选实现抛错时记录带
    模式名的日志，并抛出命名了模式的新错误（原始异常作为 ``__cause__``
    保留），让任务明确失败并归因到具体渲染器。
    """
    mode = resolve(configured_mode)
    implementation = implementations[mode]
    try:
        return implementation(**kwargs)
    except Exception as exc:
        logger.exception(f"{label} failed ({switch_name}={mode!r})")
        raise RuntimeError(
            f"{label} failed with {switch_name}={mode!r}: {exc}"
        ) from exc


def _get_required_video_duration(audio_duration: float) -> float:
    """
    返回视频素材拼接的目标时长。

    使用场景：合成视频时需要素材时长覆盖旁白音频。只做到“刚好等于”
    音频时长时，FFmpeg 可能因为帧率舍入让最终视频略短，因此统一加一个
    轻量余量。函数独立出来，便于测试和后续按实际反馈调整余量大小。
    """
    return max(0.0, float(audio_duration) + _VIDEO_DURATION_SAFETY_MARGIN)


# A3 窗口合并阈值：末窗短于该值时并入前窗，避免亚秒闪帧。A3 基准样例把
# 有效阈值约束在 (0.744, 0.816] 区间（D=3.744/W=3 末窗 0.744 → 合并为单窗；
# D=12.816/W=3 末窗 0.816 → 保留为独立短窗、维持 5 窗配额），这里取区间内
# 的 0.75 定版；floor 参数仅允许调用方进一步收紧（取 min），默认 1.0 不生效。
_WINDOW_MERGE_TAIL_SECONDS = 0.75

# 装配层段填充判定容差：每段实际放置输出时长与 segment_duration 的允许
# 偏差（帧率舍入在单段内累计不超过几十毫秒）。
_SEGMENT_FILL_TOLERANCE = 0.05


def segment_window_plan(
    segment_duration: float, max_clip_duration: float, floor: float = 1.0
) -> list[float]:
    """
    计算单个 segment 的输出窗口序列（每窗输出秒数，总和 == segment_duration）。

    A3（F-H1）决策：n = ceil(D/W)，前 n-1 窗满 W，末窗 = 余量，精确覆盖
    旁白时长。旧装配按满窗轮播，逐段超配最多 2.9s，concat 只截总时长不修
    逐段对齐，段边界漂移实测 +2.18s 累计至 +12.38s（任务 044529cb）。

    装配层（按窗放置素材）与素材层（按窗数决定下载配额，B3）共用本函数，
    两边必须看到同一个切分，配额才不会与时间线错位。
    """
    if segment_duration <= 0 or max_clip_duration <= 0:
        return []
    n = max(1, math.ceil(segment_duration / max_clip_duration))
    last = segment_duration - (n - 1) * max_clip_duration
    merge_tail = min(floor, _WINDOW_MERGE_TAIL_SECONDS)
    if n > 1 and last < merge_tail:
        # 末窗过短（闪帧级）：并入前窗，少切一刀。
        n -= 1
        last = segment_duration - (n - 1) * max_clip_duration
    return [max_clip_duration] * (n - 1) + [last]


def is_material_resolution_acceptable(width: int, height: int) -> bool:
    """
    判断素材分辨率是否足够用于合成。

    标称最小值是 480x480，但允许比它低 `_MIN_DIMENSION_TOLERANCE` 个像素，
    以兼容编码器/消息应用向下取整导致的尺寸（例如 WhatsApp 的 478x850）。
    """
    min_dimension = _MIN_MATERIAL_DIMENSION - _MIN_DIMENSION_TOLERANCE
    return width >= min_dimension and height >= min_dimension


def _prioritize_unique_source_clips(
    subclipped_items: List[SubClippedVideoClip],
    concat_mode: VideoConcatMode,
) -> List[SubClippedVideoClip]:
    """
    优先让每个源素材只出现一次，降低成片里同一素材反复出现的概率。

    线上素材经常会遇到“一个长视频被切成多个短片段”的情况。旧逻辑在
    random 模式下直接打乱所有短片段，导致同一个源视频的多个切片可能
    分布在开头和中间，用户会感知为素材重复。本函数只调整片段顺序：
    先放每个源文件里最长的一个片段，剩余片段作为兜底；当素材总时长不足时，
    仍然允许后续片段补齐音频长度，避免破坏视频生成成功率。优先选择最长
    片段是为了避免随机选中视频尾部的零碎短片段，导致明明有足够素材却过早复用。
    """
    if not subclipped_items:
        return []

    concat_mode_value = getattr(concat_mode, "value", concat_mode)
    if concat_mode_value != VideoConcatMode.random.value:
        return subclipped_items

    grouped_items: dict[str, list[SubClippedVideoClip]] = {}
    for item in subclipped_items:
        grouped_items.setdefault(item.source_file_path, []).append(item)

    primary_items = []
    overflow_items = []
    for items in grouped_items.values():
        primary_item = max(items, key=lambda item: item.duration)
        primary_items.append(primary_item)
        overflow_items.extend(item for item in items if item is not primary_item)

    random.shuffle(primary_items)
    random.shuffle(overflow_items)
    logger.info(
        "prioritized unique video materials, "
        f"sources: {len(grouped_items)}, "
        f"primary clips: {len(primary_items)}, "
        f"fallback clips: {len(overflow_items)}"
    )
    return primary_items + overflow_items


def get_ffmpeg_binary():
    """
    兼容历史上直接从 video 服务读取 FFmpeg 路径的调用方。

    真正的解析逻辑已经抽到 `app.utils.utils.get_ffmpeg_binary()`，视频、语音
    和后续新增链路都应复用同一套优先级；这里保留薄包装，避免外部脚本或
    旧测试直接导入 `app.services.video.get_ffmpeg_binary` 时出现 AttributeError。
    """
    return utils.get_ffmpeg_binary()


def _get_configured_video_codec() -> str:
    """
    读取用户配置的视频编码器。

    该配置面向高级用户，用于尝试启用 NVENC/AMF/QSV/VideoToolbox 等硬件
    编码。这里刻意只允许固定白名单，避免开放任意 FFmpeg 参数后，用户填错
    参数导致输出格式不可控，甚至让生成任务在后续阶段才失败。
    """
    configured_codec = str(
        config.app.get("video_codec", _DEFAULT_VIDEO_CODEC) or _DEFAULT_VIDEO_CODEC
    ).strip()
    if configured_codec not in _SUPPORTED_VIDEO_CODECS:
        logger.warning(
            f"unsupported video codec configured: {configured_codec}, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC
    return configured_codec


@lru_cache(maxsize=16)
def _ffmpeg_encoder_exists(ffmpeg_binary: str, codec: str) -> bool:
    """
    检查当前 FFmpeg 是否声明支持指定编码器。

    这只能证明 FFmpeg 编译时包含该 encoder，不能证明当前机器硬件和驱动
    一定可用。因此实际编码失败时仍会再回退到 libx264。
    """
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {str(exc)}"
        )
        return False

    if result.returncode != 0:
        logger.warning(
            "failed to inspect ffmpeg encoders, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}: {(result.stderr or result.stdout or '').strip()}"
        )
        return False
    return codec in result.stdout


def _get_effective_video_codec(preferred_codec: str | None = None) -> str:
    """
    返回本次实际使用的视频编码器。

    用户选择硬件编码器时，先做 FFmpeg encoder 列表检测；如果本进程里已经
    实际编码失败过，也直接回退，避免一个任务里每个片段都重复失败。
    """
    selected_codec = preferred_codec or _get_configured_video_codec()
    if selected_codec == _DEFAULT_VIDEO_CODEC:
        return _DEFAULT_VIDEO_CODEC

    if selected_codec in _runtime_disabled_video_codecs:
        logger.warning(
            f"video codec {selected_codec} was disabled after a runtime failure, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if not _ffmpeg_encoder_exists(ffmpeg_binary, selected_codec):
        logger.warning(
            f"ffmpeg encoder {selected_codec} is not available, "
            f"fallback to {_DEFAULT_VIDEO_CODEC}"
        )
        return _DEFAULT_VIDEO_CODEC

    return selected_codec


def _disable_runtime_video_codec(codec: str, reason: str):
    if codec == _DEFAULT_VIDEO_CODEC:
        return
    _runtime_disabled_video_codecs.add(codec)
    logger.warning(
        f"video codec {codec} failed, fallback to {_DEFAULT_VIDEO_CODEC}. "
        f"reason: {reason}"
    )


def _run_ffmpeg_with_codec_fallback(build_command, *, label: str):
    """
    Run an FFmpeg command, retrying once with libx264 if the chosen hardware
    encoder fails.

    This is the command-level twin of `_write_videofile_with_codec_fallback`
    for the paths that shell out to FFmpeg directly (concat, combine). It is a
    *codec* fallback, not a renderer fallback: ADR-0013 forbids silently
    switching `combine_render_mode` / `final_render_mode`, which this does not
    do — the selected renderer still runs, only the encoder changes.
    """
    effective_codec = _get_effective_video_codec()

    def run(codec: str):
        command = build_command(codec)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or f"{label} failed")
        return codec

    try:
        return run(effective_codec)
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        result_codec = run(_DEFAULT_VIDEO_CODEC)
        _disable_runtime_video_codec(effective_codec, str(exc))
        return result_codec


def _get_temp_audio_dir(output_dir: str) -> str:
    """
    Return the directory to use for MoviePy's temporary audio file.

    On Windows, Windows Defender can lock files written to the task output
    directory while scanning them, causing MoviePy to fail with a
    PermissionError (WinError 32) on the TEMP_MPY_wvf_snd temp file and
    leaving the final MP4 at 0 bytes.  Using the system temp directory
    sidesteps the scan without changing behaviour on other platforms.

    On Linux/macOS/Docker the output directory is returned unchanged so
    existing behaviour is preserved.
    """
    if sys.platform == "win32":
        return tempfile.gettempdir()
    return output_dir


def _fallback_write_videofile(clip, output_file: str, failed_codec: str, reason: str, **kwargs):
    """
    硬件编码失败后用 libx264 重试，只有重试成功才禁用该硬件编码器。

    Windows 上 FFmpeg 失败原因比较复杂：可能是显卡/驱动不支持，也可能是输出
    文件被占用、目录权限、杀软拦截等通用 IO 问题。只有 libx264 能成功写出时，
    才能判断原始失败大概率来自硬件编码器本身，避免误伤后续任务。
    """
    clip.write_videofile(output_file, codec=_DEFAULT_VIDEO_CODEC, **kwargs)
    _disable_runtime_video_codec(failed_codec, reason)
    return _DEFAULT_VIDEO_CODEC


def _write_videofile_with_codec_fallback(clip, output_file: str, codec: str, **kwargs):
    """
    使用指定编码器写出视频，失败时自动用 libx264 重试一次。

    硬件编码器是否可用不仅取决于 FFmpeg，还取决于显卡、驱动和当前运行环境。
    生成任务不能因为高级编码器不可用而整体失败，所以这里把回退集中处理。
    """
    effective_codec = _get_effective_video_codec(codec)
    try:
        clip.write_videofile(output_file, codec=effective_codec, **kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == _DEFAULT_VIDEO_CODEC:
            raise
        return _fallback_write_videofile(
            clip,
            output_file,
            failed_codec=effective_codec,
            reason=str(exc),
            **kwargs,
        )


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer 使用单引号包裹路径，路径中的单引号需要先转义。
    return file_path.replace("'", "'\\''")


def _format_ffmpeg_concat_path(file_path: str) -> str:
    """
    生成 concat demuxer 文件列表中的路径。

    FFmpeg 官方文档要求 concat list 中的特殊字符和空格需要转义；Windows
    绝对路径里的反斜杠也容易被解析成转义字符。这里统一转成正斜杠形式，
    让 `C:\\Users\\...` 变成 `C:/Users/...`，再处理单引号，兼容 macOS/Linux。
    """
    absolute_path = os.path.abspath(file_path)
    return _escape_ffmpeg_concat_path(absolute_path.replace("\\", "/"))


def concat_video_clips_with_ffmpeg(
    clip_files: List[str],
    output_file: str,
    threads: int,
    output_dir: str,
    max_duration: float | None = None,
):
    output_stem = os.path.splitext(os.path.basename(output_file))[0]
    # concat 列表按成片命名并保留在任务目录中，作为时间线拼装顺序的审计记录；
    # 与临时片段一样不再清理，任务删除时随任务目录一起回收。
    concat_list_file = os.path.join(
        output_dir, f"ffmpeg-concat-list-{output_stem}.txt"
    )
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            fp.write(f"file '{_format_ffmpeg_concat_path(clip_file)}'\n")

    def build_command(codec: str) -> list[str]:
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            "-c:v",
            codec,
            "-threads",
            str(threads or 2),
            "-pix_fmt",
            "yuv420p",
        ]
        if max_duration is not None and max_duration > 0:
            command.extend(["-t", f"{max_duration:.3f}"])
        command.append(output_file)
        return command

    # concat 列表文件按成片命名保留，作为时间线拼装顺序的审计记录，不再清理。
    # 使用 ffmpeg 只做一次串联与编码，避免 MoviePy 逐段合并时反复重编码，
    # 从而降低画质劣化与颜色偏移风险。
    return _run_ffmpeg_with_codec_fallback(
        build_command, label="ffmpeg concat"
    )


def _sanitize_image_file(image_path: str) -> str:
    # 某些本地图片虽然能被 Pillow 打开，但会因为损坏的 EXIF/eXIf 元数据导致
    # ImageClip 在解析阶段直接抛异常。这里重新导出一份“干净图片”，把坏元数据剥离掉。
    image_root, _ = os.path.splitext(image_path)
    sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        image.load()
        # 统一导出为 PNG，避免 JPEG/PNG 不同元数据路径继续把坏块带过去。
        cleaned_image = Image.new(image.mode, image.size)
        cleaned_image.putdata(list(image.getdata()))
        cleaned_image.save(sanitized_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # 优先直接打开原始图片；如果因为损坏元数据失败，再尝试生成无元数据副本。
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    安静地打开视频文件，避免 MoviePy 2.1.x 把 ffmpeg 探测信息直接打印到 stdout。

    背景：
    当前依赖版本的 `FFMPEG_VideoReader` 内部存在 `print(self.infos)` 和
    `print(ffmpeg command)`，读取无音轨的中间视频时会输出
    `audio_found: False`。这只是输入素材 metadata，不代表最终成片没有音频，
    但会误导 WebUI/终端用户以为生成失败。

    实现：
    1. 只在打开 VideoFileClip 的短窗口内重定向 stdout；
    2. 默认 `audio=False`，因为项目视频素材阶段不需要保留素材原声，
       最终音频会在 `generate_video()` 阶段统一挂载；
    3. 如果依赖库确实输出了内容，降级为 debug 日志，便于必要时排查。
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    # 循环补足视频时，同一个临时片段路径会在 FFmpeg 拼接列表中出现多次。
    # 拼接必须保留重复项，但清理只能删除一次；这里按原顺序统一去重，让所有
    # 调用方都获得幂等行为，也避免首次删除成功后连续输出 FileNotFoundError。
    unique_files = dict.fromkeys(file for file in files if file)
    for file in unique_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            # 清理动作允许文件已经不存在，例如 FFmpeg 失败路径或并发清理已经
            # 回收文件；这不是需要用户处理的问题，不应污染生成日志。
            continue
        except OSError as e:
            # 权限、只读文件系统或磁盘异常会留下真实临时文件，保留 warning
            # 便于根据具体路径和系统错误定位环境问题。
            logger.warning(f"failed to delete temporary file {file}: {str(e)}")


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        try:
            resolved_bgm_file = bgm_service.resolve_bgm_file(bgm_file)
        except ValueError as exc:
            # API 请求里的 bgm_file 来自用户输入，只允许解析到用户 BGM 或内置
            # 歌曲目录，阻止 MoviePy 读取配置、密钥等任意服务器文件。
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, error: {str(exc)}"
            )
            return ""
        return resolved_bgm_file

    if bgm_type == "random":
        files = bgm_service.list_bgm_files()
        # 当背景音乐目录为空时，直接回退为“不使用 BGM”，避免 random.choice([]) 抛异常。
        if not files:
            logger.warning("no background music files found")
            return ""
        return random.choice(files)

    return ""


# ---------------------------------------------------------------------------
# Combine phase: ffmpeg_filter renderer (R4 / #313, ADR-0013)
#
# The MoviePy combine path writes one `temp-clip-*.mp4` per placement through
# MoviePy (GIL-held frame compositing + one encode each) and then concatenates
# them with a second encode. `ffmpeg_filter` builds the exact same timeline as
# a single `filter_complex` graph: one decode seek per source, scale/pad to the
# target aspect, optional per-clip transition, then one native `concat` and one
# encode. Clip *selection* is reproduced draw-for-draw from the MoviePy path so
# the two renderers choose the same material order; the graph reproduces
# scale-to-fit + centre-on-black (never crop) and the one-second transitions.
# ---------------------------------------------------------------------------

# Every per-clip transition is applied for one second.
_COMBINE_TRANSITION_SECONDS = 1.0
_TRANSITION_SIDES = ("left", "right", "top", "bottom")
_SLIDE_TRANSITIONS = ("SlideIn", "SlideOut")
_ZOOM_TRANSITIONS = ("ZoomIn", "ZoomOut")
_SHUFFLE_TRANSITIONS = (
    "FadeIn",
    "FadeOut",
    "SlideIn",
    "SlideOut",
    "ZoomIn",
    "ZoomOut",
)


@dataclass(frozen=True)
class _CombineClipPlan:
    """
    One placement on the combine timeline, before any encoder runs.

    ``source_path`` empty means a black placeholder (missing material or an
    explicit segment hole). ``source_start``/``source_end`` bound the read
    window in the source video, ``output_duration`` is the intended length on
    the output timeline after ``speed`` is applied, and
    ``transition``/``transition_side`` describe the per-clip effect to
    reproduce (``None`` means no effect).
    """

    source_path: str
    source_start: float
    source_end: float
    output_duration: float
    speed: float
    transition: str | None = None
    transition_side: str | None = None
    # Safety-net truncation bound applied by `_normalize_segment_clip_with_transition`; the
    # legacy path caps at `max_clip_duration`, the segment path at
    # `max(max_clip_duration, window_seconds)`.
    truncate_to: float | None = None


def _choose_transition(transition_value) -> tuple[str | None, str | None]:
    """
    Resolve a configured transition into one concrete effect identity.

    This is the single source of truth for transition selection: the MoviePy
    renderer (`_normalize_segment_clip_with_transition` / `_apply_transition`)
    and the FFmpeg graph (`_combine_transition_filters`) both switch on the
    returned identity instead of maintaining their own enum branches.

    The random draw order matches the original MoviePy behaviour: a slide side
    is drawn for *every* clip (even when no transition is configured), and for
    ``shuffle`` a second draw picks the effect. Preserving the draw order keeps
    the random concat order comparable between the two renderers.
    """
    side = random.choice(_TRANSITION_SIDES)
    value = getattr(transition_value, "value", transition_value)
    if value in (None, VideoTransitionMode.none.value):
        return None, None
    if value == VideoTransitionMode.fade_in.value:
        return "FadeIn", None
    if value == VideoTransitionMode.fade_out.value:
        return "FadeOut", None
    if value == VideoTransitionMode.slide_in.value:
        return "SlideIn", side
    if value == VideoTransitionMode.slide_out.value:
        return "SlideOut", side
    if value == VideoTransitionMode.zoom_in.value:
        return "ZoomIn", None
    if value == VideoTransitionMode.zoom_out.value:
        return "ZoomOut", None
    if value == VideoTransitionMode.shuffle.value:
        chosen = random.choice(_SHUFFLE_TRANSITIONS)
        return chosen, (side if chosen in _SLIDE_TRANSITIONS else None)
    # Unknown values are silently ignored by MoviePy too; mirror that.
    return None, None


def _probe_video_duration(video_path: str) -> float:
    """Read a source clip's duration without decoding any frames."""
    clip = _open_video_clip_quietly(video_path)
    try:
        return float(clip.duration or 0.0)
    finally:
        close_clip(clip)


def _iter_legacy_placements(
    video_paths: List[str],
    *,
    required_video_duration: float,
    max_clip_duration: float,
    video_concat_mode,
    clip_speed: float,
    transition_value,
):
    """
    Yield the timeline placements for the random/sequential combine path.

    Single source of truth for both renderers: `_combine_videos_moviepy`
    consumes it and renders each yielded clip, `_plan_combine_clips_legacy`
    consumes it for the FFmpeg graph. It only *selects* — probing source
    durations (one open per source, matching the historical first loop) and
    choosing transitions; the renderer opens the source for the actual clip.
    """
    normalized_speed = utils.normalize_clip_speed(clip_speed)
    # max_clip_duration 约束的是成片里的最终播放时长，而不是源视频读取时长。
    # 以 0.5 倍速播放 1.5 秒源画面会得到 3 秒片段，以 2 倍速播放 6 秒源画面
    # 同样得到 3 秒片段。因此切片前必须按速度反推源时长，保证不同速度下源
    # 时间线连续且无重叠。
    source_clip_duration = max_clip_duration * normalized_speed
    concat_value = getattr(video_concat_mode, "value", video_concat_mode)

    subclipped_items: List[SubClippedVideoClip] = []
    for video_path in video_paths:
        clip_duration = _probe_video_duration(video_path)
        start_time = 0.0
        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        source_file_path=video_path,
                    )
                )
            start_time = end_time
            if concat_value == VideoConcatMode.sequential.value:
                break

    subclipped_items = _prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
    )
    logger.debug(f"total subclipped items: {len(subclipped_items)}")

    total = 0.0
    base_plans: List[_CombineClipPlan] = []
    for item in subclipped_items:
        if total >= required_video_duration:
            break
        source_length = item.end_time - item.start_time
        output_duration = min(
            source_length / normalized_speed if normalized_speed else source_length,
            max_clip_duration,
        )
        transition, side = _choose_transition(transition_value)
        plan = _CombineClipPlan(
            source_path=item.file_path,
            source_start=item.start_time,
            source_end=item.end_time,
            output_duration=output_duration,
            speed=normalized_speed,
            transition=transition,
            transition_side=side,
            truncate_to=max_clip_duration,
        )
        base_plans.append(plan)
        total += output_duration
        yield plan

    # Loop the already-selected clips until the narration is covered.
    if total < required_video_duration and base_plans:
        for plan in itertools.cycle(list(base_plans)):
            if total >= required_video_duration:
                break
            total += plan.output_duration
            yield plan


def _plan_combine_clips_legacy(
    video_paths: List[str],
    *,
    required_video_duration: float,
    max_clip_duration: float,
    video_concat_mode,
    clip_speed: float,
    transition_value,
) -> tuple[List[_CombineClipPlan], float]:
    """Collect `_iter_legacy_placements` into a plan list (FFmpeg path)."""
    plans = list(
        _iter_legacy_placements(
            video_paths,
            required_video_duration=required_video_duration,
            max_clip_duration=max_clip_duration,
            video_concat_mode=video_concat_mode,
            clip_speed=clip_speed,
            transition_value=transition_value,
        )
    )
    return plans, sum(plan.output_duration for plan in plans)


def _iter_segment_placements(
    segments: List[dict],
    *,
    max_clip_duration: float,
    clip_speed: float,
    transition_value,
    advance_clip_window: bool = True,
    dedupe_clips_across_segments: bool = True,
    retain_clip: bool = True,
):
    """
    Yield `(plan, clip)` pairs for the segment-first combine path.

    Single source of truth for both renderers. ``retain_clip=True`` yields the
    opened, sub-clipped, speed-adjusted MoviePy clip for the renderer to write
    (the open is shared with the duration read, so the historical one-open-per-
    placement contract is preserved); ``retain_clip=False`` closes the clip and
    yields ``None`` for the FFmpeg graph, which only needs the plan.
    """
    normalized_speed = utils.normalize_clip_speed(clip_speed)
    used_clip_paths: set[str] = set()

    for segment in segments:
        clip_paths = list(segment.get("clips") or [])
        if dedupe_clips_across_segments:
            fresh = [p for p in clip_paths if p not in used_clip_paths]
            reused = [p for p in clip_paths if p in used_clip_paths]
            clip_paths = fresh + reused

        if not clip_paths:
            placeholder_duration = float(segment.get("duration") or 0)
            if placeholder_duration <= 0:
                continue
            yield (
                _CombineClipPlan(
                    source_path="",
                    source_start=0.0,
                    source_end=0.0,
                    output_duration=placeholder_duration,
                    speed=1.0,
                    truncate_to=placeholder_duration,
                ),
                None,
            )
            continue

        segment_duration = float(segment.get("duration") or 0)
        windows = segment_window_plan(segment_duration, max_clip_duration)
        segment_remaining = (
            segment_duration if segment_duration > 0 else max_clip_duration
        )
        if not windows:
            windows = [max_clip_duration]

        clip_cycle = itertools.cycle(clip_paths)
        window_offset: dict[str, float] = {}
        plan_index = 0
        plan_intact = True
        hole_slots = set(segment.get("holes") or [])

        while segment_remaining > _SEGMENT_FILL_TOLERANCE:
            if plan_index < len(windows) and plan_intact:
                window_seconds = windows[plan_index]
            else:
                window_seconds = min(max_clip_duration, segment_remaining)
            plan_index += 1

            if (plan_index - 1) in hole_slots:
                segment_remaining -= window_seconds
                yield (
                    _CombineClipPlan(
                        source_path="",
                        source_start=0.0,
                        source_end=0.0,
                        output_duration=window_seconds,
                        speed=1.0,
                        truncate_to=window_seconds,
                    ),
                    None,
                )
                continue

            video_path = next(clip_cycle)
            start_offset = (
                window_offset.get(video_path, 0.0) if advance_clip_window else 0.0
            )
            try:
                clip = _open_video_clip_quietly(video_path)
                source_duration = float(clip.duration or 0)
                if (
                    advance_clip_window
                    and source_duration > 0
                    and start_offset >= source_duration
                ):
                    start_offset = 0.0
                target_source = window_seconds * normalized_speed
                if source_duration > 0:
                    available_source = min(
                        target_source, max(source_duration - start_offset, 0.0)
                    )
                    clip = clip.subclipped(
                        start_offset, start_offset + available_source
                    )
                else:
                    available_source = float(clip.duration or 0)
                if normalized_speed != 1.0:
                    clip = clip.with_speed_scaled(normalized_speed)
            except Exception as exc:
                logger.error(
                    "failed to process segment clip: "
                    f"segment={segment.get('index')}, file={video_path}, error: {exc}"
                )
                break

            placed = float(clip.duration or 0)
            if not retain_clip:
                close_clip(clip)
                clip_out = None
            else:
                clip_out = clip

            if advance_clip_window and source_duration > 0:
                next_offset = start_offset + available_source
                window_offset[video_path] = (
                    0.0 if next_offset >= source_duration else next_offset
                )
            if dedupe_clips_across_segments:
                used_clip_paths.add(video_path)
            transition, side = _choose_transition(transition_value)
            segment_remaining -= placed
            if placed < window_seconds - 0.01:
                plan_intact = False

            yield (
                _CombineClipPlan(
                    source_path=video_path,
                    source_start=start_offset,
                    source_end=start_offset + available_source,
                    output_duration=placed,
                    speed=normalized_speed,
                    transition=transition,
                    transition_side=side,
                    truncate_to=max(max_clip_duration, window_seconds),
                ),
                clip_out,
            )


def _plan_combine_clips_segment_first(
    segments: List[dict],
    *,
    max_clip_duration: float,
    clip_speed: float,
    transition_value,
    advance_clip_window: bool = True,
    dedupe_clips_across_segments: bool = True,
) -> tuple[List[_CombineClipPlan], float]:
    """Collect `_iter_segment_placements` into a plan list (FFmpeg path)."""
    plans = [
        plan
        for plan, _clip in _iter_segment_placements(
            segments,
            max_clip_duration=max_clip_duration,
            clip_speed=clip_speed,
            transition_value=transition_value,
            advance_clip_window=advance_clip_window,
            dedupe_clips_across_segments=dedupe_clips_across_segments,
            retain_clip=False,
        )
    ]
    return plans, sum(plan.output_duration for plan in plans)


def _combine_transition_filters(
    plan: _CombineClipPlan,
    index: int,
    width: int,
    height: int,
    video_fps: int,
) -> List[str]:
    """Return the filter chain fragments for one clip's configured transition."""
    transition = plan.transition
    if transition in ("FadeIn", "FadeOut"):
        fade_in = transition == "FadeIn"
        start = 0.0 if fade_in else max(0.0, plan.output_duration - _COMBINE_TRANSITION_SECONDS)
        kind = "in" if fade_in else "out"
        return [
            f"[s{index}]fade=t={kind}:st={start:.3f}:d={_COMBINE_TRANSITION_SECONDS:.3f}"
            f"[v{index}]"
        ]
    if transition in _SLIDE_TRANSITIONS:
        # Reproduce the explicit black-background + positional animation used by
        # `video_effects.slidein_transition` / `slideout_transition`.
        if transition == "SlideIn":
            progress = "min(t/1,1)"
        else:
            progress = f"max(0,min((t-({plan.output_duration:.3f}-1))/1,1))"
        side = plan.transition_side or "left"
        slide_in = transition == "SlideIn"
        if side == "left":
            x = (
                f"'-{width}+{width}*{progress}'"
                if slide_in
                else f"'-{width}*{progress}'"
            )
            y = "0"
        elif side == "right":
            x = (
                f"'{width}-{width}*{progress}'"
                if slide_in
                else f"'{width}*{progress}'"
            )
            y = "0"
        elif side == "top":
            y = (
                f"'-{height}+{height}*{progress}'"
                if slide_in
                else f"'-{height}*{progress}'"
            )
            x = "0"
        else:  # bottom
            y = (
                f"'{height}-{height}*{progress}'"
                if slide_in
                else f"'{height}*{progress}'"
            )
            x = "0"
        return [
            f"color=c=black:s={width}x{height}:r={video_fps},"
            f"trim=duration={plan.output_duration:.3f}[bg{index}]",
            f"[bg{index}][s{index}]overlay=x={x}:y={y}:"
            f"eof_action=pass:shortest=0[v{index}]",
        ]
    if transition in _ZOOM_TRANSITIONS:
        frames = max(1, int(round(plan.output_duration * video_fps)))
        if transition == "ZoomIn":
            zoom_expr = f"1+0.2*on/{frames}"
        else:
            zoom_expr = f"1.2-0.2*on/{frames}"
        return [
            f"[s{index}]zoompan=z='{zoom_expr}':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:s={width}x{height}:fps={video_fps}[v{index}]"
        ]
    return [f"[s{index}]null[v{index}]"]


def _build_combine_ffmpeg_command(
    plans: List[_CombineClipPlan],
    output_file: str,
    *,
    width: int,
    height: int,
    video_fps: int,
    threads: int,
    codec: str,
    max_duration: float | None = None,
) -> List[str]:
    """Build the single `filter_complex` command for a combine timeline plan."""
    input_args: List[str] = []
    chains: List[str] = []

    for index, plan in enumerate(plans):
        if plan.source_path:
            span = max(0.0, plan.source_end - plan.source_start)
            input_args += [
                "-ss",
                f"{max(0.0, plan.source_start):.3f}",
                "-t",
                f"{span:.3f}",
                "-i",
                plan.source_path,
            ]
            pre_filters: List[str] = []
            if abs(plan.speed - 1.0) > 1e-9:
                pre_filters.append(f"setpts=PTS/{plan.speed:.6f}")
            pre_filters.append(
                f"trim=duration={plan.output_duration:.3f},setpts=PTS-STARTPTS"
            )
            pre = ",".join(pre_filters) + ","
        else:
            input_args += [
                "-f",
                "lavfi",
                "-t",
                f"{plan.output_duration:.3f}",
                "-i",
                f"color=c=black:s={width}x{height}:r={video_fps}",
            ]
            pre = ""

        chains.append(
            f"[{index}:v]{pre}fps={video_fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,setsar=1,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black[s{index}]"
        )
        chains.extend(
            _combine_transition_filters(plan, index, width, height, video_fps)
        )

    # concat does not auto-convert inside filter_complex: normalise every arm
    # (transition outputs included) to yuv420p before joining them.
    normalise = [
        f"[v{index}]format=yuv420p[c{index}]" for index in range(len(plans))
    ]
    concat_inputs = "".join(f"[c{index}]" for index in range(len(plans)))
    graph = ";".join(
        chains
        + normalise
        + [f"{concat_inputs}concat=n={len(plans)}:v=1:a=0[vout]"]
    )

    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        *input_args,
        "-filter_complex",
        graph,
        "-map",
        "[vout]",
        "-c:v",
        codec,
        "-threads",
        str(threads or 2),
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(video_fps),
    ]
    if max_duration is not None and max_duration > 0:
        command += ["-t", f"{max_duration:.3f}"]
    # The combine stage is video-only; narration is mixed in by generate_video.
    command += ["-an", output_file]
    return command


def _combine_videos_ffmpeg_filter(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    segments: List[dict] | None = None,
    advance_clip_window: bool = True,
    dedupe_clips_across_segments: bool = True,
) -> str:
    """
    Combine phase renderer that emits one `filter_complex` pass (R4 / #313).

    Selected explicitly through ``combine_render_mode = "ffmpeg_filter"``;
    there is no runtime fallback. Clip selection, fit/pad, speed and per-clip
    transitions are reproduced from the MoviePy path, but the whole timeline is
    decoded/composited/encoded once by FFmpeg instead of via per-clip MoviePy
    temp files.
    """
    audio_clip = AudioFileClip(audio_file)
    try:
        audio_duration = float(audio_clip.duration or 0.0)
    finally:
        close_clip(audio_clip)
    logger.info(f"ffmpeg_filter combine: audio duration: {audio_duration} seconds")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)

    if segments:
        plans, planned_duration = _plan_combine_clips_segment_first(
            segments,
            max_clip_duration=max_clip_duration,
            clip_speed=clip_speed,
            transition_value=transition_value,
            advance_clip_window=advance_clip_window,
            dedupe_clips_across_segments=dedupe_clips_across_segments,
        )
    else:
        required_video_duration = _get_required_video_duration(audio_duration)
        plans, planned_duration = _plan_combine_clips_legacy(
            video_paths,
            required_video_duration=required_video_duration,
            max_clip_duration=max_clip_duration,
            video_concat_mode=video_concat_mode,
            clip_speed=clip_speed,
            transition_value=transition_value,
        )

    if not plans:
        logger.warning("no clips available for merging (ffmpeg_filter)")
        return combined_video_path

    logger.info(
        f"ffmpeg_filter combine: {len(plans)} placements, "
        f"{planned_duration:.2f}s planned, one filter_complex pass"
    )

    def build_command(codec: str):
        return _build_combine_ffmpeg_command(
            plans,
            combined_video_path,
            width=video_width,
            height=video_height,
            video_fps=fps,
            threads=threads,
            codec=codec,
            max_duration=audio_duration,
        )

    _run_ffmpeg_with_codec_fallback(
        build_command, label="ffmpeg combined render (ffmpeg_filter)"
    )
    # Same contract as `_combine_videos_moviepy`: return the combined path, not
    # the codec (the codec fallback helper returns the codec it used).
    return combined_video_path


def _combine_videos_moviepy(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    segments: List[dict] | None = None,
    advance_clip_window: bool = True,
    dedupe_clips_across_segments: bool = True,
) -> str:
    if segments:
        return _combine_videos_segment_first(
            combined_video_path=combined_video_path,
            segments=segments,
            audio_file=audio_file,
            video_aspect=video_aspect,
            video_transition_mode=video_transition_mode,
            max_clip_duration=max_clip_duration,
            threads=threads,
            clip_speed=clip_speed,
            advance_clip_window=advance_clip_window,
            dedupe_clips_across_segments=dedupe_clips_across_segments,
        )

    audio_clip = AudioFileClip(audio_file)
    try:
        # 这里只需要读取旁白音频时长来决定素材视频拼接长度；后续不会再使用
        # audio_clip。读取完成后立即关闭，避免早退或异常路径泄漏文件句柄。
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")
    required_video_duration = _get_required_video_duration(audio_duration)
    logger.info(
        f"required video duration: {required_video_duration:.2f} seconds "
        f"(audio duration + {_VIDEO_DURATION_SAFETY_MARGIN:.2f}s safety margin)"
    )

    # 兼容 API 直接调用时未传转场模式的情况，避免后续访问 .value 时崩溃。
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        # 只记录一次最终生效值，既方便定位 API 越界参数被归一化的问题，
        # 也避免在逐片段热路径中重复输出相同日志。
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    rendered_clips = {}
    video_duration = 0.0

    for i, plan in enumerate(
        _iter_legacy_placements(
            video_paths,
            required_video_duration=required_video_duration,
            max_clip_duration=max_clip_duration,
            video_concat_mode=video_concat_mode,
            clip_speed=clip_speed,
            transition_value=transition_value,
        )
    ):
        logger.debug(
            f"processing clip {i + 1}: source: "
            f"{os.path.basename(plan.source_path)}, "
            f"current duration: {video_duration:.2f}s, "
            f"remaining: {required_video_duration - video_duration:.2f}s"
        )

        # The generator replays placements to cover the narration; the MoviePy
        # path keeps reusing the already-rendered temp file for a replay rather
        # than re-encoding it (same behaviour as the old `processed_clips`
        # cycle).
        cache_key = (
            plan.source_path,
            round(plan.source_start, 4),
            round(plan.source_end, 4),
            round(plan.speed, 4),
            plan.transition,
            plan.transition_side,
        )
        replayed = rendered_clips.get(cache_key)
        if replayed is not None:
            processed_clips.append(replayed)
            video_duration += replayed.duration
            continue

        try:
            clip = _open_video_clip_quietly(plan.source_path).subclipped(
                plan.source_start, plan.source_end
            )
            # 播放速度属于素材本身属性，应在转场前应用。这样 Fade/Slide 等一秒转场
            # 不会跟随素材速度变成 0.5 秒或 2 秒；后续最大时长裁剪继续作为
            # 浮点误差或异常素材时长的安全兜底，保证最终片段不突破配置上限。
            if plan.speed != 1.0:
                clip = clip.with_speed_scaled(plan.speed)
            # 缩放/转场/时长兜底与 segment-first 路径共用同一实现，
            # 避免两条流水线的画面行为出现差异。转场身份已由共享生成器解析。
            clip = _normalize_segment_clip_with_transition(
                clip,
                video_width,
                video_height,
                plan.truncate_to if plan.truncate_to is not None else max_clip_duration,
                plan.transition,
                plan.transition_side,
            )
            clip_w, clip_h = clip.size

            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i + 1}.mp4"
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            entry = SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=clip_w,
                height=clip_h,
                source_file_path=plan.source_path,
            )
            rendered_clips[cache_key] = entry
            processed_clips.append(entry)
            video_duration += clip_duration_saved

        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")

    if video_duration < required_video_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than required duration "
            f"({required_video_duration:.2f}s) after rendering."
        )

    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        return combined_video_path

    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
        max_duration=audio_duration,
    )

    # 临时片段（temp-clip-*.mp4）保留在任务目录中供审计时间线来源，不再清理；
    # 任务删除时随任务目录一起回收。
    logger.info(f"preserved {len(clip_files)} intermediate clip files for audit")

    logger.info("video combining completed")
    return combined_video_path


_COMBINE_RENDER_IMPLEMENTATIONS = {
    COMBINE_RENDER_MODE_MOVIEPY: _combine_videos_moviepy,
    COMBINE_RENDER_MODE_FFMPEG_FILTER: _combine_videos_ffmpeg_filter,
}


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    segments: List[dict] | None = None,
    advance_clip_window: bool = True,
    dedupe_clips_across_segments: bool = True,
) -> str:
    """
    素材拼接阶段入口：按 ``combine_render_mode`` 显式分派到唯一实现。

    选择的结果就是唯一执行的路径，没有运行时回退。未知取值在启动校验时
    已被拒绝；这里再做一次防御式解析，保证直接调用服务层（CLI/脚本）时
    也遵循同一契约。
    """
    return _dispatch_renderer(
        switch_name="combine_render_mode",
        configured_mode=config.app.get("combine_render_mode"),
        resolve=resolve_combine_render_mode,
        implementations=_COMBINE_RENDER_IMPLEMENTATIONS,
        label="combine phase",
        combined_video_path=combined_video_path,
        video_paths=video_paths,
        audio_file=audio_file,
        video_aspect=video_aspect,
        video_concat_mode=video_concat_mode,
        video_transition_mode=video_transition_mode,
        max_clip_duration=max_clip_duration,
        threads=threads,
        clip_speed=clip_speed,
        segments=segments,
        advance_clip_window=advance_clip_window,
        dedupe_clips_across_segments=dedupe_clips_across_segments,
    )


def _fit_clip_to_canvas(clip, video_width: int, video_height: int):
    """
    把素材等比缩放到目标画幅，不足处用黑边补齐（只 fit，不 crop）。

    `combine_videos` 的旧流程和 segment-first 路径共用；`ffmpeg_filter`
    渲染器的 scale/pad 图节点是这段逻辑的等价实现，两者必须保持一致。
    """
    clip_duration = clip.duration
    clip_w, clip_h = clip.size
    if clip_w == video_width and clip_h == video_height:
        return clip

    clip_ratio = clip.w / clip.h
    video_ratio = video_width / video_height
    logger.debug(
        f"resizing clip, source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, "
        f"target: {video_width}x{video_height}, ratio: {video_ratio:.2f}"
    )

    if clip_ratio == video_ratio:
        return clip.resized(new_size=(video_width, video_height))

    if clip_ratio > video_ratio:
        scale_factor = video_width / clip_w
    else:
        scale_factor = video_height / clip_h

    new_width = int(clip_w * scale_factor)
    new_height = int(clip_h * scale_factor)

    background = ColorClip(
        size=(video_width, video_height), color=(0, 0, 0)
    ).with_duration(clip_duration)
    clip_resized = clip.resized(new_size=(new_width, new_height)).with_position(
        "center"
    )
    return CompositeVideoClip([background, clip_resized])


def _apply_transition(clip, transition: str | None, side: str | None):
    """
    把 `_choose_transition` 解析出的转场应用到 MoviePy clip。

    转场身份的单一事实来源是 `_choose_transition`；本函数只负责把身份映射
    到 `video_effects` 的具体实现，`_combine_transition_filters` 则把同一
    身份映射到等价的 FFmpeg filter，两者不会再各自维护一套枚举分支。
    """
    if transition == "FadeIn":
        return video_effects.fadein_transition(clip, _COMBINE_TRANSITION_SECONDS)
    if transition == "FadeOut":
        return video_effects.fadeout_transition(clip, _COMBINE_TRANSITION_SECONDS)
    if transition == "SlideIn":
        return video_effects.slidein_transition(
            clip, _COMBINE_TRANSITION_SECONDS, side or "left"
        )
    if transition == "SlideOut":
        return video_effects.slideout_transition(
            clip, _COMBINE_TRANSITION_SECONDS, side or "left"
        )
    if transition == "ZoomIn":
        return video_effects.zoomin_transition(clip, _COMBINE_TRANSITION_SECONDS)
    if transition == "ZoomOut":
        return video_effects.zoomout_transition(clip, _COMBINE_TRANSITION_SECONDS)
    return clip


def _normalize_segment_clip_with_transition(
    clip,
    video_width: int,
    video_height: int,
    truncate_to: float,
    transition: str | None,
    side: str | None,
):
    """
    Fit/transition/truncate a clip using an already-resolved transition.

    Used by the placement consumers: the transition identity comes from the
    shared generator's `_choose_transition`, so this must not draw again.
    """
    clip = _fit_clip_to_canvas(clip, video_width, video_height)
    clip = _apply_transition(clip, transition, side)
    if clip.duration > truncate_to:
        clip = clip.subclipped(0, truncate_to)
    return clip


def _combine_videos_segment_first(
    combined_video_path: str,
    segments: List[dict],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    advance_clip_window: bool = True,
    dedupe_clips_across_segments: bool = True,
) -> str:
    """
    按 segment 顺序拼接已对齐的素材片段（segment-first 路径）。

    与随机拼接不同，这里没有任何打乱逻辑：segments 的顺序就是旁白顺序，
    每个片段来源都由任务编排层按 segment 搜索得到，因此拼接结果天然与
    旁白对齐。单个 segment 缺少素材时跳过（音频仍连续），整段缺失素材
    时仅记录警告，最终成片时长由旁白音频决定。

    advance_clip_window（默认开启）：同一源视频被同一 segment 的轮播再次
    选中时，截取窗口按 max_clip_duration 依次后移（0-3s、3-6s…），而不是
    每次都取前 3 秒，消除同源内容的重复画面。

    dedupe_clips_across_segments（默认开启）：segment 之间共享一个"本任务已
    使用"的源视频集合，各 segment 优先选用未出现过的候选，候选全部用过
    时才回退复用，避免热门素材在相邻 segment 反复出现。
    """
    audio_clip = AudioFileClip(audio_file)
    try:
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(
        f"segment-first assembly: audio duration: {audio_duration} seconds, "
        f"segments: {len(segments)}"
    )

    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    normalized_clip_speed = utils.normalize_clip_speed(clip_speed)
    if normalized_clip_speed != 1.0:
        logger.info(f"clip playback speed: {normalized_clip_speed:.2f}x")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()
    output_dir = os.path.dirname(combined_video_path)

    processed_clips: List[SubClippedVideoClip] = []
    clip_sequence = 0

    for plan, clip in _iter_segment_placements(
        segments,
        max_clip_duration=max_clip_duration,
        clip_speed=clip_speed,
        transition_value=transition_value,
        advance_clip_window=advance_clip_window,
        dedupe_clips_across_segments=dedupe_clips_across_segments,
        retain_clip=True,
    ):
        clip_file = f"{output_dir}/temp-clip-{clip_sequence + 1}.mp4"
        clip_sequence += 1

        if plan.source_path == "":
            # 时间线对齐要求每段的画面时长覆盖该段旁白时长。没有素材或显式
            # backfill hole 时用黑屏占位而不是跳过；占位片段不做转场，与旧
            # 行为一致。
            placeholder = ColorClip(
                size=(video_width, video_height), color=(0, 0, 0)
            ).with_duration(plan.output_duration)
            placeholder.write_videofile(clip_file, fps=fps, logger=None)
            close_clip(placeholder)
            processed_clips.append(
                SubClippedVideoClip(
                    file_path=clip_file,
                    duration=plan.output_duration,
                    width=video_width,
                    height=video_height,
                    source_file_path="",
                )
            )
            continue

        if clip is None:
            raise RuntimeError(
                f"segment-first placement missing clip: segment="
                f"{plan.source_path}@{plan.source_start:.3f}"
            )

        try:
            clip = _normalize_segment_clip_with_transition(
                clip,
                video_width,
                video_height,
                plan.truncate_to
                if plan.truncate_to is not None
                else max(max_clip_duration, plan.output_duration),
                plan.transition,
                plan.transition_side,
            )
            _write_videofile_with_codec_fallback(
                clip,
                clip_file,
                codec=_get_configured_video_codec(),
                logger=None,
                fps=fps,
            )
            clip_duration_saved = clip.duration
        finally:
            close_clip(clip)

        processed_clips.append(
            SubClippedVideoClip(
                file_path=clip_file,
                duration=clip_duration_saved,
                width=video_width,
                height=video_height,
                source_file_path=plan.source_path,
            )
        )

    logger.info("starting segment clip merging process")
    if not processed_clips:
        logger.warning("no segment clips available for merging")
        return combined_video_path

    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} segment clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
        max_duration=audio_duration,
    )

    # 临时片段（temp-clip-*.mp4）保留在任务目录中供审计每段画面的实际拼装
    # 顺序与来源，不再清理；任务删除时随任务目录一起回收。
    logger.info(f"preserved {len(clip_files)} intermediate segment clip files for audit")
    logger.info("segment-first video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # 字幕换行必须在真正创建 TextClip 前完成，否则 MoviePy 只会按原始文本
    # 计算渲染区域。这里用 PIL 按当前字体和字号测量宽度，确保每一行都尽量
    # 控制在视频可用宽度内，避免大字号或中文长句直接溢出画面。
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, fontsize
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    def split_long_token(token):
        # 当一个 token 本身就超宽时（常见于中文无空格长句，或英文超长单词），
        # 退化为字符级拆分。关键点是：检测到 candidate 超宽时，先提交上一个
        # 仍然合法的 current，再把当前字符放入下一行，不能把超宽字符塞回上一行。
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    line_start_punctuation = "，。！？；：、,.!?;:)]}）】》」』”’"
    for index in range(1, len(lines)):
        # 中文长句按字符拆分时，最后一个句号、逗号等闭合标点可能被单独
        # 放到下一行，导致字幕背景被异常撑高，视觉上像一个小点掉在正文
        # 下方。这里在不重新设计换行算法的前提下，把上一行最后一个字
        # 移到标点行前面，让标点跟随文字显示，兼容中英文常见闭合标点。
        if not lines[index] or lines[index][0] not in line_start_punctuation:
            continue
        if len(lines[index - 1]) <= 1:
            continue

        candidate = f"{lines[index - 1][-1]}{lines[index]}"
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            lines[index] = candidate
            lines[index - 1] = lines[index - 1][:-1]

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    height = len(lines) * height
    return result, height


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # 字幕背景色来自 API/WebUI 参数，可能为空或格式不规范。这里统一只接受
    # #RRGGBB 形式，非法值回退为黑色，避免 PIL 渲染阶段抛出异常中断任务。
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            pass
    return (0, 0, 0)


def _rounded_subtitle_background_clip(
    width: int,
    height: int,
    color: str,
    alpha: int = 140,
    radius: int = 16,
) -> ImageClip:
    # 新字幕背景仅在用户显式开启时使用：通过 RGBA 图片绘制圆角半透明底板，
    # 再交给 MoviePy 作为透明 ImageClip 参与合成。这样默认路径完全不变，
    # 同时可以低成本试验更柔和的字幕视觉效果。
    rgb = _hex_to_rgb(color)
    safe_alpha = max(0, min(255, int(alpha)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, max(0, width - 1), max(0, height - 1)],
        radius=max(0, int(radius)),
        fill=(rgb[0], rgb[1], rgb[2], safe_alpha),
    )
    return ImageClip(np.array(img), transparent=True)


def _get_visible_center_position(
    text_clip: TextClip,
    container_width: int,
    container_height: int,
) -> tuple[int, int]:
    """
    按文字真实可见像素把 TextClip 放到背景容器中心。

    MoviePy 的 TextClip 会按字体行高和 baseline 创建透明画布。很多字体的
    可见字形并不在这个画布的几何中心，直接 `with_position("center")`
    会把整块透明画布居中，导致字幕看起来偏上或偏下。这里读取 TextClip
    的透明 mask，只根据实际有像素的 bbox 计算偏移，让用户看到的文字
    在字幕背景里视觉居中。
    """
    x = int(round((container_width - text_clip.w) / 2))
    y = int(round((container_height - text_clip.h) / 2))

    try:
        if text_clip.mask is None:
            return x, y

        mask_frame = text_clip.mask.get_frame(0)
        ys, _ = np.where(mask_frame > 0.01)
        if len(ys) == 0:
            return x, y

        visible_top = int(ys.min())
        visible_bottom = int(ys.max())
        visible_height = visible_bottom - visible_top + 1
        y = int(round((container_height - visible_height) / 2 - visible_top))
    except Exception as exc:
        logger.debug(f"failed to center subtitle text by visible mask: {str(exc)}")

    return x, y


def subtitle_colors_are_indistinguishable(params: VideoParams) -> bool:
    """判断字幕文字和背景是否同色，提醒用户可能无法看清字幕。"""
    if not params.subtitle_enabled or not params.text_background_color:
        return False

    def normalize_color(value):
        if isinstance(value, bool):
            return "#000000" if value else ""
        return str(value or "").strip().lower()

    text_color = normalize_color(params.text_fore_color)
    background_color = normalize_color(params.text_background_color)
    return bool(text_color and text_color == background_color)


@lru_cache(maxsize=64)
def _subtitle_font_supports_sample(font_path: str, sample: str) -> bool:
    """检查字体是否包含样本文字需要的字形，并缓存重复检查结果。"""
    try:
        font = ImageFont.truetype(font_path, 30)
        missing_mask = font.getmask("\U0010ffff")
        missing_signature = (
            missing_mask.size,
            missing_mask.getbbox(),
            bytes(missing_mask),
        )
        for char in sample:
            char_mask = font.getmask(char)
            char_signature = (
                char_mask.size,
                char_mask.getbbox(),
                bytes(char_mask),
            )
            if char_mask.getbbox() is None or char_signature == missing_signature:
                return False
        return True
    except Exception as e:
        # 字体探测失败不应阻止用户生成；保留日志供环境兼容问题排查。
        logger.warning(f"failed to inspect subtitle font glyphs: {font_path}, {e}")
        return True


def subtitle_font_supports_text(font_path: str, text: str) -> bool:
    """检查字体能否绘制文本中的字母和数字，忽略空白及标点符号。"""
    sample = "".join(
        dict.fromkeys(
            char
            for char in str(text or "")
            if unicodedata.category(char)[0] in {"L", "N"}
        )
    )[:64]
    if not sample:
        return True
    return _subtitle_font_supports_sample(font_path, sample)


@dataclass(frozen=True)
class _SubtitleOverlay:
    """
    单个字幕短语的预渲染图层。

    ``rgba`` 是字幕背景板 + 文字已经合成好的 RGBA 数组（uint8），尺寸只覆盖
    字幕实际占用的小块区域；``x``/``y`` 是它相对成片画布左上角的落点。
    """

    start: float
    end: float
    x: int
    y: int
    rgba: np.ndarray


def _build_subtitle_overlay(clip, canvas_size) -> _SubtitleOverlay:
    """
    把一个已定位的字幕 clip 预渲染成 RGBA 图层并缓存。

    字幕 clip 是静态的（不随时间变化），所以这里只在构建阶段调用一次
    ``get_frame``/``mask``；以后每帧都复用同一份 RGBA 数组，避免 MoviePy
    在每帧里重复做整帧 RGBA ``astype`` + ``alpha_composite``（F2 热点）。
    """
    frame = np.asarray(clip.get_frame(0))
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim == 2:
        frame = np.dstack([frame, frame, frame])
    frame = np.ascontiguousarray(frame[:, :, :3])

    if clip.mask is not None:
        alpha = (clip.mask.get_frame(0) * 255).astype(np.uint8)
    else:
        alpha = np.full(frame.shape[:2], 255, dtype=np.uint8)

    rgba = np.ascontiguousarray(np.dstack([frame, alpha]))

    # 与 MoviePy `compose_on` 使用同一套位置解析，保证落点逐像素一致。
    pos = compute_position(clip.size, canvas_size, clip.pos(0), clip.relative_pos)
    end = clip.end if clip.end is not None else clip.duration
    return _SubtitleOverlay(
        start=float(clip.start or 0.0),
        end=float(end),
        x=int(pos[0]),
        y=int(pos[1]),
        rgba=rgba,
    )


def _blit_rgba_overlay(frame: np.ndarray, overlay: _SubtitleOverlay) -> None:
    """
    把预渲染字幕图层原地叠加到 ``frame``（uint8 RGB）上，只处理包围盒。

    使用 PIL ``alpha_composite`` 而不是手写整数公式，是为了与 MoviePy
    原来的合成结果保持逐像素一致：alpha 合成是逐像素操作，包围盒之外的
    图层 alpha 为 0，因此裁到包围盒不会改变结果。
    """
    rgba = overlay.rgba
    x, y = overlay.x, overlay.y
    overlay_h, overlay_w = rgba.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    x_start = max(x, 0)
    y_start = max(y, 0)
    x_end = min(x + overlay_w, frame_w)
    y_end = min(y + overlay_h, frame_h)
    if x_end <= x_start or y_end <= y_start:
        return

    source = rgba[y_start - y : y_end - y, x_start - x : x_end - x]
    region = frame[y_start:y_end, x_start:x_end]
    composited = Image.alpha_composite(
        Image.fromarray(region).convert("RGBA"),
        Image.fromarray(np.ascontiguousarray(source)),
    )
    frame[y_start:y_end, x_start:x_end] = np.asarray(composited)[:, :, :3]


class _SubtitleOverlayClip(VideoClip):
    """
    把预渲染好的字幕图层按包围盒叠加到基础视频上的合成 clip。

    与 ``CompositeVideoClip([video, *text_clips])`` 视觉结果一致，但每帧只
    在字幕覆盖的像素区域内做 alpha 合成，并全程停留在 uint8，避免 MoviePy
    每帧整帧 RGBA 合成带来的 CPU 开销（瓶颈 F2）。
    """

    def __init__(self, base_clip, overlays: List[_SubtitleOverlay]):
        self.base_clip = base_clip
        self.overlays = list(overlays)
        # 与旧的 `CompositeVideoClip` 一致：时长取所有子剪辑 end 的最大值，
        # 而不是只取基础视频时长，避免字幕尾部超出视频时被提前截断。
        overlay_end = max((overlay.end for overlay in self.overlays), default=0.0)
        duration = max(base_clip.duration or 0.0, overlay_end)
        super().__init__(
            frame_function=self._compose_frame,
            duration=duration,
        )
        self.size = base_clip.size
        self.fps = getattr(base_clip, "fps", None)

    def _compose_frame(self, t: float) -> np.ndarray:
        frame = np.array(self.base_clip.get_frame(t), dtype=np.uint8, copy=True)
        for overlay in self.overlays:
            if overlay.start <= t < overlay.end:
                _blit_rgba_overlay(frame, overlay)
        return frame

    def close(self):
        # base_clip 由调用方的 ExitStack 负责关闭；这里释放字幕图层引用，
        # 并调用父类 close（Clip.close 目前为空实现，但保持覆盖链完整）。
        self.overlays = []
        super().close()


def _generate_video_moviepy(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    合成最终视频，并返回本次背景音乐处理是否成功。

    返回值只描述 BGM 处理状态：没有请求 BGM 或成功混合时返回 True；请求了
    BGM 但加载、特效或混合失败时返回 False。即使 BGM 失败仍会继续输出只有
    旁白的视频，让任务编排层决定是否向用户展示降级警告。
    """
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def resolve_subtitle_background_color():
        # 兼容历史参数：API 里 `text_background_color` 既可能是布尔值，
        # 也可能是实际颜色字符串。统一在这里归一化，避免把 True/False
        # 直接传给 TextClip 后出现不可预期的渲染结果。
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        bg_color = resolve_subtitle_background_color()
        rounded_bg_enabled = bool(
            getattr(params, "rounded_subtitle_background", False) and bg_color
        )
        has_subtitle_background = bool(bg_color)
        # 圆角背景按文字真实宽度生成，左右留白应更克制；旧矩形背景仍保留
        # 较大的安全边距，避免历史配置中的长字幕贴边或被裁切。
        padding_ratio = 0.4 if rounded_bg_enabled else 0.6
        pad_x = int(params.font_size * padding_ratio) if has_subtitle_background else 0
        # 字幕背景需要给文字左右留出明确内边距。先从可用宽度中扣除
        # padding 再换行，避免长英文或大字号刚好撑满 90% 视频宽度后，
        # 文字贴到背景框边缘，看起来像被裁切。普通矩形背景和圆角背景
        # 都走这条逻辑；无背景字幕则保持原有最大宽度。
        text_max_width = max(1, int(max_width) - 2 * pad_x)
        wrapped_txt, txt_height = wrap_text(
            phrase,
            max_width=text_max_width,
            font=font_path,
            fontsize=params.font_size,
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        text_clip_margin_y = max(
            int(params.font_size * 0.3), int(params.stroke_width * 2)
        )
        # MoviePy 在 `method=label` 下会自动收缩文本框高度，遇到多行字幕、
        # 描边或背景色时，容易把最后一行的下半部分裁掉。这里显式传入
        # 一个更保守的高度，把行间距和额外上下留白一并算进去，保证字幕
        # 背景框与文字本身都能完整渲染出来。
        clip_h = int(txt_height + vertical_padding + (interline * line_count))

        if rounded_bg_enabled:
            # 圆角背景需要贴合文字宽度，而不是沿用 90% 视频宽度。这里先用
            # PIL 测量最长一行文字，再加水平内边距，避免短字幕出现过宽底板。
            try:
                font = ImageFont.truetype(font_path, params.font_size)
                text_w = max(
                    int(font.getbbox(line)[2] - font.getbbox(line)[0])
                    for line in wrapped_txt.split("\n")
                )
            except Exception as exc:
                logger.warning(
                    f"failed to measure subtitle text width, fallback to max width: {str(exc)}"
                )
                text_w = int(max_width)

            box_w = max(1, min(int(max_width), text_w + 2 * pad_x))
            radius = max(8, int(params.font_size * 0.4))
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(box_w, None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            clip_h = max(clip_h, text_clip.h)
            bg_clip = _rounded_subtitle_background_clip(
                width=box_w,
                height=clip_h,
                color=bg_color,
                alpha=140,
                radius=radius,
            )
            text_position = _get_visible_center_position(text_clip, box_w, clip_h)
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=(box_w, clip_h),
            )
        elif bg_color:
            size = (
                int(max_width),
                clip_h,
            )
            text_clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=(int(max_width), None),
                text_align="center",
                margin=(0, text_clip_margin_y),
            )
            size = (size[0], max(size[1], text_clip.h))
            bg_clip = _rounded_subtitle_background_clip(
                width=size[0],
                height=size[1],
                color=bg_color,
                alpha=255,
                radius=0,
            )
            text_position = _get_visible_center_position(text_clip, size[0], size[1])
            _clip = CompositeVideoClip(
                [bg_clip, text_clip.with_position(text_position)],
                size=size,
            )
        else:
            size = (
                int(max_width),
                clip_h,
            )
            _clip = TextClip(
                text=wrapped_txt,
                font=font_path,
                font_size=params.font_size,
                color=params.text_fore_color,
                bg_color=None,
                stroke_color=params.stroke_color,
                stroke_width=params.stroke_width,
                interline=interline,
                size=size,
                text_align="center",
            )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)
        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    # MoviePy 的 CompositeAudioClip.close() 不会关闭子 AudioFileClip。这里用
    # ExitStack 显式持有所有原始文件 reader，确保成功、字幕异常、混音失败和
    # 视频写入失败等路径都能释放 FFmpeg 子进程，尤其避免 Windows 文件被占用。
    with ExitStack() as clip_stack:
        source_video_clip = clip_stack.enter_context(
            _open_video_clip_quietly(video_path)
        )
        voice_source_clip = clip_stack.enter_context(AudioFileClip(audio_path))
        video_clip = source_video_clip
        audio_clip = voice_source_clip.with_effects(
            [afx.MultiplyVolume(params.voice_volume)]
        )

        def make_textclip(text):
            return TextClip(
                text=text,
                font=font_path,
                font_size=params.font_size,
            )

        if subtitle_path and os.path.exists(subtitle_path):
            sub = clip_stack.enter_context(
                SubtitlesClip(
                    subtitles=subtitle_path,
                    encoding="utf-8",
                    make_textclip=make_textclip,
                )
            )
            # F2（#311）：每个字幕短语只预渲染一次，合成时只按包围盒叠加，
            # 避免 MoviePy 每帧对整帧做 RGBA astype/alpha_composite。
            overlays = [
                _build_subtitle_overlay(
                    create_text_clip(subtitle_item=item), video_clip.size
                )
                for item in sub.subtitles
            ]
            video_clip = _SubtitleOverlayClip(video_clip, overlays)
            clip_stack.callback(video_clip.close)

        bgm_enabled = bgm_service.should_use_bgm(
            params.bgm_type, params.bgm_volume
        )
        if not bgm_enabled and params.bgm_type:
            # 所有 BGM 来源共用这一条短路规则。音量不大于 0 时不能解析随机或
            # 自定义文件，也不能加载提供商返回的文件，避免无意义的 IO 和混音。
            logger.info(
                f"skipping background music because volume is not positive: "
                f"type={params.bgm_type}, volume={params.bgm_volume}"
            )

        # 提供商配乐可由任务编排层直接传入对应文件。None 表示沿用随机/自定义
        # BGM 解析，空字符串明确禁用本条 BGM；但任何来源都必须先通过通用音量规则。
        bgm_file = ""
        if bgm_enabled:
            bgm_file = (
                bgm_file_override
                if bgm_file_override is not None
                else get_bgm_file(
                    bgm_type=params.bgm_type,
                    bgm_file=params.bgm_file,
                )
            )
        bgm_mix_succeeded = True
        if bgm_file:
            try:
                bgm_effects = [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                ]
                # 服务内解析的随机/自定义音乐可能比成片短，需要循环铺满；任务层
                # 通过 override 传入的文件表示提供商已经完成时长适配。这里依据
                # 文件来源决定是否循环，避免今后每增加一个提供商都修改名称白名单。
                if bgm_file_override is None:
                    bgm_effects.append(afx.AudioLoop(duration=video_clip.duration))
                bgm_source_clip = clip_stack.enter_context(AudioFileClip(bgm_file))
                bgm_clip = bgm_source_clip.with_effects(bgm_effects)
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception:
                bgm_mix_succeeded = False
                # 记录完整堆栈和稳定上下文，便于区分文件解码、MoviePy 特效和
                # CompositeAudioClip 失败；文件内容与 API Key 不会进入日志。
                logger.exception(
                    f"failed to mix background music: type={params.bgm_type}, "
                    f"file={bgm_file}"
                )

        final_video_clip = video_clip.with_audio(audio_clip)
        clip_stack.callback(final_video_clip.close)
        # 显式沿用输入音频的采样率；如果取不到，再回退 MoviePy 默认的 44100Hz。
        # 这样可以减少不同环境，尤其 Docker 中再次重采样带来的音质波动。
        output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
        _write_videofile_with_codec_fallback(
            final_video_clip,
            output_file=output_file,
            codec=_get_configured_video_codec(),
            audio_codec=audio_codec,
            audio_fps=output_audio_fps,
            audio_bitrate=audio_bitrate,
            temp_audiofile_path=_get_temp_audio_dir(output_dir),
            threads=params.n_threads or 2,
            logger=None,
            fps=fps,
        )
        return bgm_mix_succeeded


_FINAL_RENDER_IMPLEMENTATIONS = {
    FINAL_RENDER_MODE_MOVIEPY: _generate_video_moviepy,
}


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    """
    最终成片入口：按 ``final_render_mode`` 显式分派到唯一实现。

    选择的结果就是唯一执行的路径，没有运行时回退。未知取值在启动校验时
    已被拒绝；这里再做一次防御式解析，保证直接调用服务层（CLI/脚本）时
    也遵循同一契约。
    """
    return _dispatch_renderer(
        switch_name="final_render_mode",
        configured_mode=config.app.get("final_render_mode"),
        resolve=resolve_final_render_mode,
        implementations=_FINAL_RENDER_IMPLEMENTATIONS,
        label="final render",
        video_path=video_path,
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        output_file=output_file,
        params=params,
        bgm_file_override=bgm_file_override,
    )


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    # WebUI 在某些二次生成场景下可能传入空素材列表，这里直接返回空结果，避免抛出 NoneType 异常。
    if not materials:
        return []

    # 仅返回通过预处理校验的素材，避免低分辨率图片继续进入后续的视频合成流程。
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # local video_source 的素材路径来自 API 参数，必须限制在专用素材目录。
            # 允许用户传文件名，也兼容历史返回的绝对路径，但不允许逃逸到系统
            # 其他目录，避免任意文件读取或通过 MoviePy 探测本地敏感文件。
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        try:
            # 图片素材直接按图片方式读取，避免先走 VideoFileClip 误判后触发不稳定的回退分支。
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # 非标准扩展名或探测失败时再回退到图片模式，兼容历史上直接传本地图片路径的情况。
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if not is_material_resolution_acceptable(width, height):
                logger.warning(
                    f"low resolution material: {width}x{height}, minimum "
                    f"{_MIN_MATERIAL_DIMENSION}x{_MIN_MATERIAL_DIMENSION} required "
                    f"(tolerance {_MIN_DIMENSION_TOLERANCE}px)"
                )
                # 探测到低分辨率素材后立即关闭资源，并且不要把该素材返回给后续流程。
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # 探测尺寸时已经打开过一次素材，这里先释放探测句柄，再重新创建用于导出的图片 clip。
                close_clip(clip)
                # Create an image clip and set its duration to 3 seconds
                clip = (
                    ImageClip(material_source_path)
                    .with_duration(clip_duration)
                    .with_position("center")
                )
                # Apply a zoom effect using the resize method.
                # A lambda function is used to make the zoom effect dynamic over time.
                # The zoom effect starts from the original size and gradually scales up to 120%.
                # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
                # Note: 1 represents 100% size, so 1.2 represents 120% size.
                zoom_clip = clip.resized(
                    lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
                )

                # Optionally, create a composite video clip containing the zoomed clip.
                # This is useful when you want to add other elements to the video.
                final_clip = CompositeVideoClip([zoom_clip])

                # Output the video to a file.
                video_file = f"{material_source_path}.mp4"
                final_clip.write_videofile(video_file, fps=30, logger=None)
                close_clip(clip)
                close_clip(final_clip)
                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # 普通视频素材只需要读取尺寸做校验，校验完成后立即释放句柄即可。
                close_clip(clip)
                # Update url to the resolved absolute path so that downstream
                # stages (combine_videos) can open the file without re-resolving.
                material.url = material_source_path
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials
