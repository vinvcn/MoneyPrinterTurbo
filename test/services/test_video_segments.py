import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import video as vd


class _FakeAudioClip:
    def __init__(self, duration):
        self.duration = duration

    def close(self):
        pass


class _FakeVideoClip:
    def __init__(self, duration):
        self.duration = duration
        self.size = (1080, 1920)
        self.w = 1080
        self.h = 1920

    def subclipped(self, start_time, end_time):
        return _FakeVideoClip(end_time - start_time)

    def with_speed_scaled(self, factor):
        return _FakeVideoClip(self.duration / factor)

    def close(self):
        pass


class TestCombineSegmentVideos(unittest.TestCase):
    def _run_combine_segments(
        self, segments, video_clips, audio_duration=20.0, **combine_kwargs
    ):
        """Run combine_videos with timed segments; capture concat order + cut windows."""
        concat_calls = []

        def fake_concat(clip_files, output_file, threads, output_dir, max_duration=None):
            concat_calls.append(list(clip_files))

        written_clips = []

        def fake_open(video_path, audio=False):
            return _FakeVideoClip(video_clips[video_path])

        def fake_write(clip, output_file, **kwargs):
            written_clips.append(output_file)
            # _normalize_segment_clip 之前已 subclipped；用 clip.duration 记录
            # 每个窗口实际渲染的时长，配合 opened 顺序还原窗口起点。
            written_clips[-1] = (output_file, clip.duration)
            Path(output_file).write_bytes(b"clip")

        opened_order = []

        def opening_spy(video_path, audio=False):
            opened_order.append(video_path)
            return _FakeVideoClip(video_clips[video_path])

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"fake")
            combined_path = str(Path(temp_dir) / "combined.mp4")
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip(audio_duration)),
                patch.object(vd, "_open_video_clip_quietly", side_effect=opening_spy),
                patch.object(vd, "_write_videofile_with_codec_fallback", side_effect=fake_write),
                patch.object(vd, "concat_video_clips_with_ffmpeg", side_effect=fake_concat),
            ):
                vd.combine_videos(
                    combined_video_path=combined_path,
                    video_paths=[],
                    audio_file=str(audio_path),
                    segments=segments,
                    **combine_kwargs,
                )
            return concat_calls, opened_order, written_clips

    def test_segments_drive_concat_order(self):
        """Clips must be concatenated in segment order — no shuffle."""
        segments = [
            {"index": 0, "clips": ["/c/a.mp4", "/c/b.mp4"], "duration": 10.0},
            {"index": 1, "clips": ["/c/c.mp4"], "duration": 10.0},
        ]
        video_clips = {"/c/a.mp4": 10.0, "/c/b.mp4": 10.0, "/c/c.mp4": 10.0}
        concat_calls, _, _ = self._run_combine_segments(segments, video_clips)

        self.assertEqual(len(concat_calls), 1)
        written = [Path(p).name for p in concat_calls[0]]
        # Segment 0 contributes 2 clips (each up to 5s covering 10s), then segment 1.
        self.assertTrue(all(os.path.isabs(p) or p for p in written))
        # Temp file order encodes processing order; segment 1's clip must
        # come after segment 0's.
        self.assertLess(written.index("temp-clip-1.mp4"), written.index("temp-clip-3.mp4"))

    def test_empty_clips_get_black_placeholder(self):
        """无素材片段用黑屏占位，保证后续片段不前移、旁白保持对齐。"""
        segments = [{"index": 0, "clips": [], "duration": 10.0}]
        concat_calls, _, _ = self._run_combine_segments(segments, {})
        self.assertEqual(len(concat_calls), 1)
        # 占位片段也被拼接（这里 concat 与写出都被 mock，只验证时间线包含它）。
        self.assertEqual(concat_calls[0][0].split("/")[-1], "temp-clip-1.mp4")

    def test_empty_clips_without_duration_are_skipped(self):
        segments = [{"index": 0, "clips": [], "duration": 0}]
        concat_calls, _, _ = self._run_combine_segments(segments, {})
        self.assertEqual(concat_calls, [])


class TestAdvanceClipWindow(TestCombineSegmentVideos):
    """advance_clip_window：同源视频轮播再次选中时，窗口后移而不是重复前 3 秒。"""

    def test_window_advances_on_reuse_within_segment(self):
        # 12s 源视频、3s 窗口被同一 segment 轮询 3 次 → 起点 0、3、6。
        segments = [
            {"index": 0, "clips": ["/c/a.mp4"], "duration": 9.0},
        ]
        video_clips = {"/c/a.mp4": 12.0}
        _, opened, written = self._run_combine_segments(
            segments, video_clips, max_clip_duration=3
        )

        # 3 次打开同一源，每次渲染 3s。
        self.assertEqual(len(opened), 3)
        self.assertTrue(all(p == "/c/a.mp4" for p in opened))
        self.assertEqual([dur for _, dur in written], [3.0, 3.0, 3.0])

    def test_window_wraps_around_when_source_exhausted(self):
        # 4s 源视频、3s 窗口：第一次 0-3s，第二次 3-4s（1s），第三次回绕 0-3s。
        segments = [
            {"index": 0, "clips": ["/c/short.mp4"], "duration": 7.0},
        ]
        video_clips = {"/c/short.mp4": 4.0}
        _, opened, written = self._run_combine_segments(
            segments, video_clips, max_clip_duration=3
        )

        self.assertEqual(len(opened), 3)
        durations = [dur for _, dur in written]
        # 3s + 1s + 3s = 7s 覆盖旁白。
        self.assertEqual(durations, [3.0, 1.0, 3.0])

    def test_window_advance_disabled_reuses_first_window(self):
        """开关关闭时保持旧行为：每次都取前 3 秒。"""
        segments = [
            {"index": 0, "clips": ["/c/a.mp4"], "duration": 9.0},
        ]
        video_clips = {"/c/a.mp4": 12.0}
        # advance 不可通过 combine_videos 直接断言窗口内容（subclipped 在
        # _normalize_segment_clip 前），用 opened/written 数量 + duration
        # 无法区分起点；这里通过 monkeypatch subclipped 记录起点。
        import app.services.video as vd_module

        starts = []

        class _SpyClip(_FakeVideoClip):
            def subclipped(self, start_time, end_time):
                starts.append(start_time)
                return _FakeVideoClip(end_time - start_time)

        def fake_open(video_path, audio=False):
            return _SpyClip(video_clips[video_path])

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"fake")
            combined_path = str(Path(temp_dir) / "combined.mp4")
            with (
                patch.object(vd_module, "AudioFileClip", return_value=_FakeAudioClip(20.0)),
                patch.object(vd_module, "_open_video_clip_quietly", side_effect=fake_open),
                patch.object(vd_module, "_write_videofile_with_codec_fallback"),
                patch.object(vd_module, "concat_video_clips_with_ffmpeg"),
            ):
                vd_module.combine_videos(
                    combined_video_path=combined_path,
                    video_paths=[],
                    audio_file=str(audio_path),
                    segments=segments,
                    max_clip_duration=3,
                    advance_clip_window=False,
                )
        self.assertEqual(starts, [0.0, 0.0, 0.0])


class TestDedupeClipsAcrossSegments(TestCombineSegmentVideos):
    """dedupe_clips_across_segments：后续 segment 优先使用未上过时间线的候选。"""

    def test_shared_source_deferred_behind_fresh_ones(self):
        # seg0 用 [a, b]，seg1 候选 [a, c] → a 被延后，seg1 先用 c。
        segments = [
            {"index": 0, "clips": ["/c/a.mp4", "/c/b.mp4"], "duration": 6.0},
            {"index": 1, "clips": ["/c/a.mp4", "/c/c.mp4"], "duration": 6.0},
        ]
        video_clips = {"/c/a.mp4": 10.0, "/c/b.mp4": 10.0, "/c/c.mp4": 10.0}
        _, opened, _ = self._run_combine_segments(segments, video_clips)

        # seg0: a, b (6s/3s=2)；seg1: c, a（c 未用过先选）。
        self.assertEqual(opened, ["/c/a.mp4", "/c/b.mp4", "/c/c.mp4", "/c/a.mp4"])

    def test_dedupe_disabled_keeps_candidate_order(self):
        segments = [
            {"index": 0, "clips": ["/c/a.mp4", "/c/b.mp4"], "duration": 6.0},
            {"index": 1, "clips": ["/c/a.mp4", "/c/c.mp4"], "duration": 6.0},
        ]
        video_clips = {"/c/a.mp4": 10.0, "/c/b.mp4": 10.0, "/c/c.mp4": 10.0}
        _, opened, _ = self._run_combine_segments(
            segments, video_clips, dedupe_clips_across_segments=False
        )
        # 关闭去重：seg1 按原候选顺序 a, c。
        self.assertEqual(opened, ["/c/a.mp4", "/c/b.mp4", "/c/a.mp4", "/c/c.mp4"])


if __name__ == "__main__":
    unittest.main()
