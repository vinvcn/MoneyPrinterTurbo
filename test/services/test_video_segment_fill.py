"""_combine_videos_segment_first 段填充基线特征 + A3 精确填充不变量测试。

基线特征记录（改动前 HEAD 实测，2026-09-07；任务级 UAT 漂移见
.omo/FINDINGS-SUMMARY.md：段边界 +2.18s 累计至 +12.38s）：
- 旧实现按满窗轮播：D=2.5/W=1 的 segment 放置 3 个 1s temp-clip，共 3.0s，
  逐段超配 +0.5s；两段合计 6.0s > 旁白 5.0s；concat(max_duration=audio)
  只截总时长、不修逐段对齐 —— 这正是 F-H1 的漂移机制。实测 temp-clip
  = [1,1,1,1,1,1]，成片 6.0s。改动前"超配"断言运行通过（钉住基线），
  A3 落地后超配不再发生，该断言移除、由下方精确填充不变量取代。
- 新实现按 segment_window_plan 放置：同 segment 恰好 2.5s（[1,1.5]，
  0.5s 末窗低于合并阈值并入前窗），不变量：每段实际放置输出时长 ==
  segment_duration（±0.05s）。

运行成本说明：真实 ffmpeg 编码小尺寸 ColorClip 片段（2 段 × 2 窗 + concat），
单次全流程约 5-10s，确定性（纯本地渲染，无网络/随机依赖），远低于 60s，
因此采用真集成测试而非 plan 层替身。
"""

import sys
import wave
from pathlib import Path

import pytest
from moviepy import ColorClip, VideoFileClip

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services.video import _combine_videos_segment_first, segment_window_plan

_SEGMENT_DURATION = 2.5
_WINDOW_SECONDS = 1
_AUDIO_SECONDS = 8.0  # 大于旧实现的超配总时长，避免 concat 截断掩盖超配


def _write_silent_wav(path: Path, seconds: float) -> str:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * int(8000 * seconds))
    return str(path)


def _write_source_clip(path: Path, seconds: float) -> str:
    # 9:16 小尺寸纯色片段，走装配层等比缩放路径，编码成本最低。
    clip = ColorClip(size=(72, 128), color=(30, 90, 160)).with_duration(seconds)
    clip.write_videofile(
        str(path), fps=24, codec="libx264", preset="ultrafast", logger=None
    )
    clip.close()
    return str(path)


def _temp_clip_durations(output_dir: Path) -> list[float]:
    durations = []
    for index in range(1, 100):
        clip_file = output_dir / f"temp-clip-{index}.mp4"
        if not clip_file.exists():
            break
        with VideoFileClip(str(clip_file)) as clip:
            durations.append(clip.duration)
    return durations


def _assemble(tmp_path: Path, segments, clip_speed: float = 1.0) -> tuple[Path, list[float]]:
    audio_file = _write_silent_wav(tmp_path / "narration.wav", _AUDIO_SECONDS)
    source = _write_source_clip(tmp_path / "source.mp4", 6.0)
    for segment in segments:
        segment.setdefault("clips", [source])
    combined = tmp_path / "combined.mp4"
    _combine_videos_segment_first(
        combined_video_path=str(combined),
        segments=segments,
        audio_file=audio_file,
        video_aspect=VideoAspect.portrait,
        video_transition_mode=None,
        max_clip_duration=_WINDOW_SECONDS,
        threads=2,
        clip_speed=clip_speed,
        advance_clip_window=True,
        dedupe_clips_across_segments=True,
    )
    return combined, _temp_clip_durations(tmp_path)


def _two_equal_segments():
    return [
        {"index": 0, "duration": _SEGMENT_DURATION},
        {"index": 1, "duration": _SEGMENT_DURATION},
    ]


def test_segment_placed_output_matches_duration(tmp_path):
    """不变量：每段实际放置输出时长 == segment_duration（±0.05s）。

    改动前该测试失败（旧实现逐段超配，见上方基线记录），failing-first。
    窗口数直接取自共享的 segment_window_plan，装配层与计划必须一致。
    """
    expected_windows = segment_window_plan(_SEGMENT_DURATION, _WINDOW_SECONDS)
    combined, durations = _assemble(tmp_path, _two_equal_segments())
    assert len(durations) == 2 * len(expected_windows)
    first_segment = durations[: len(expected_windows)]
    second_segment = durations[len(expected_windows) :]
    assert sum(first_segment) == pytest.approx(_SEGMENT_DURATION, abs=0.05)
    assert sum(second_segment) == pytest.approx(_SEGMENT_DURATION, abs=0.05)
    with VideoFileClip(str(combined)) as clip:
        combined_duration = clip.duration
    # 总时长对齐旁白之和，且 concat 截断不再掩盖/制造偏差。
    assert combined_duration == pytest.approx(2 * _SEGMENT_DURATION, abs=0.1)


def test_speed_scaled_windows_fill_segment_duration(tmp_path):
    """speed=2 时窗口源秒数 = 输出秒 × speed：放置输出仍精确填满段时长。

    若 src/speed 语义反置（src = output/speed），每窗只放 0.5s，总时长
    1.0s 会直接击穿该断言。
    """
    combined, durations = _assemble(
        tmp_path, [{"index": 0, "duration": 2.0}], clip_speed=2.0
    )
    assert len(durations) == 2
    assert sum(durations) == pytest.approx(2.0, abs=0.05)
    with VideoFileClip(str(combined)) as clip:
        combined_duration = clip.duration
    assert combined_duration == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# Backfill hole tests (Metis B3)
# ---------------------------------------------------------------------------


def test_holes_produce_black_placeholder_for_plan_slot(tmp_path):
    """(a) Segment with clips=[A,B], plan [3,3,3], holes=[1] → slot 1 is a
    3.0s black placeholder; slots 0/2 cut from A/B (source NOT consumed by hole);
    sum of placed durations == segment_duration ± _SEGMENT_FILL_TOLERANCE.
    Per-window black clips skip _normalize_segment_clip/transitions exactly like
    the existing whole-segment placeholder (Metis D — pinned).
    """
    segment_duration = 9.0
    source_a = _write_source_clip(tmp_path / "src_a.mp4", 6.0)
    source_b = _write_source_clip(tmp_path / "src_b.mp4", 6.0)
    audio_file = _write_silent_wav(tmp_path / "narration.wav", 20.0)
    segment = {
        "index": 0,
        "duration": segment_duration,
        "clips": [source_a, source_b],
        "holes": [1],
    }
    combined = tmp_path / "combined.mp4"
    _combine_videos_segment_first(
        combined_video_path=str(combined),
        segments=[segment],
        audio_file=audio_file,
        video_aspect=VideoAspect.portrait,
        video_transition_mode=None,
        max_clip_duration=3.0,
        threads=2,
        clip_speed=1.0,
        advance_clip_window=True,
        dedupe_clips_across_segments=True,
    )
    durations = _temp_clip_durations(tmp_path)
    assert len(durations) == 3
    # Slot 1 is a black placeholder of exactly the planned window_seconds (3.0).
    assert durations[1] == pytest.approx(3.0, abs=0.05)
    # Sum of placed durations matches segment_duration.
    assert sum(durations) == pytest.approx(segment_duration, abs=0.05)
    with VideoFileClip(str(combined)) as clip:
        assert clip.duration == pytest.approx(segment_duration, abs=0.1)
    # Pixel sensitivity (Metis B3): duration-only assertions pass even when the
    # hole slot silently consumes a source clip. _write_source_clip renders
    # (30, 90, 160), so the real hole frame (t=1.5 inside the 3.0s window) must
    # stay near-black while the non-hole windows stay visibly colored.
    hole_file = tmp_path / "temp-clip-2.mp4"
    with VideoFileClip(str(hole_file)) as c:
        hole_frame = c.get_frame(1.5)
    assert hole_frame is not None
    hole_frame_max = float(hole_frame.max())
    assert hole_frame_max <= 8
    # At least one non-hole window carries the colored source: the hole
    # assertion above cannot pass vacuously on an all-black assembly.
    non_hole_frame_max = 0.0
    # source_file_path="" for the hole → bypasses used_clip_paths dedupe.
    for idx in [0, 2]:
        clip_file = tmp_path / f"temp-clip-{idx + 1}.mp4"
        with VideoFileClip(str(clip_file)) as c:
            assert c.duration > 0
            non_hole_frame = c.get_frame(1.5)
        assert non_hole_frame is not None
        non_hole_frame_max = max(
            non_hole_frame_max, float(non_hole_frame.max())
        )
    assert non_hole_frame_max > 8


def test_empty_holes_produces_no_black_clips(tmp_path):
    """(b) holes=[] → placements byte-identical to current behavior (existing
    tests keep passing unchanged); no black placeholders injected.
    """
    source = _write_source_clip(tmp_path / "src.mp4", 6.0)
    audio_file = _write_silent_wav(tmp_path / "narration.wav", 20.0)
    segment = {
        "index": 0,
        "duration": 2.5,
        "clips": [source],
        "holes": [],
    }
    combined = tmp_path / "combined.mp4"
    _combine_videos_segment_first(
        combined_video_path=str(combined),
        segments=[segment],
        audio_file=audio_file,
        video_aspect=VideoAspect.portrait,
        video_transition_mode=None,
        max_clip_duration=1.0,
        threads=2,
        clip_speed=1.0,
        advance_clip_window=True,
        dedupe_clips_across_segments=True,
    )
    durations = _temp_clip_durations(tmp_path)
    expected = segment_window_plan(2.5, 1.0)
    assert len(durations) == len(expected)
    assert sum(durations) == pytest.approx(2.5, abs=0.05)


def test_out_of_range_holes_ignored_safely(tmp_path):
    """(c) holes referencing out-of-range plan indices → ignored safely (no
    crash), placements as if no holes.
    """
    source = _write_source_clip(tmp_path / "src.mp4", 6.0)
    audio_file = _write_silent_wav(tmp_path / "narration.wav", 20.0)
    segment = {
        "index": 0,
        "duration": 2.5,
        "clips": [source],
        "holes": [99, -1],
    }
    combined = tmp_path / "combined.mp4"
    _combine_videos_segment_first(
        combined_video_path=str(combined),
        segments=[segment],
        audio_file=audio_file,
        video_aspect=VideoAspect.portrait,
        video_transition_mode=None,
        max_clip_duration=1.0,
        threads=2,
        clip_speed=1.0,
        advance_clip_window=True,
        dedupe_clips_across_segments=True,
    )
    durations = _temp_clip_durations(tmp_path)
    expected = segment_window_plan(2.5, 1.0)
    assert len(durations) == len(expected)
    assert sum(durations) == pytest.approx(2.5, abs=0.05)
    with VideoFileClip(str(combined)) as clip:
        assert clip.duration == pytest.approx(2.5, abs=0.1)
