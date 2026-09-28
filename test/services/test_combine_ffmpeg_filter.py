"""
#313 / R4: the `ffmpeg_filter` combine renderer.

Covers the pure planning/command-building logic (unit) and a rendered-output
parity check against the MoviePy path (integration, needs FFmpeg).
"""

import os
import random
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from moviepy import ColorClip, VideoFileClip

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect, VideoConcatMode, VideoTransitionMode  # noqa: E402
from app.services import video as vd  # noqa: E402


def _silent_wav(path, seconds=2.0, sample_rate=44100):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def _make_clip(path, color, seconds=1.0, size=(64, 96), video_fps=10):
    clip = ColorClip(size=size, color=color).with_duration(seconds)
    clip.write_videofile(path, codec="libx264", fps=video_fps, audio=False, logger=None)
    clip.close()


class TestCombineCommandBuilder(unittest.TestCase):
    """The graph must reproduce selection, fit/pad, speed and transitions."""

    def _plan(self, **kwargs):
        defaults = dict(
            source_path="a.mp4",
            source_start=0.0,
            source_end=1.0,
            output_duration=1.0,
            speed=1.0,
        )
        defaults.update(kwargs)
        return vd._CombineClipPlan(**defaults)

    def test_source_inputs_use_seek_and_span(self):
        command = vd._build_combine_ffmpeg_command(
            [self._plan(source_start=0.5, source_end=1.5)],
            "out.mp4",
            width=1080,
            height=1920,
            video_fps=30,
            threads=4,
            codec="libx264",
        )
        self.assertIn("-ss", command)
        self.assertEqual(command[command.index("-ss") + 1], "0.500")
        self.assertEqual(command[command.index("-t") + 1], "1.000")
        self.assertEqual(command[command.index("-i") + 1], "a.mp4")
        self.assertIn("-an", command)
        self.assertEqual(command[-1], "out.mp4")

    def test_placeholder_uses_black_lavfi_source(self):
        command = vd._build_combine_ffmpeg_command(
            [self._plan(source_path="", output_duration=2.0)],
            "out.mp4",
            width=1080,
            height=1920,
            video_fps=30,
            threads=2,
            codec="libx264",
        )
        self.assertIn("lavfi", command)
        lavfi_index = command.index("-i") + 1
        self.assertIn("color=c=black", command[lavfi_index])

    def test_filter_graph_concat_and_pad(self):
        plans = [self._plan(), self._plan(source_path="", output_duration=0.5)]
        command = vd._build_combine_ffmpeg_command(
            plans,
            "out.mp4",
            width=1080,
            height=1920,
            video_fps=30,
            threads=2,
            codec="libx264",
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=decrease", graph)
        self.assertIn("pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black", graph)
        self.assertIn("concat=n=2:v=1:a=0[vout]", graph)
        self.assertEqual(command[command.index("-map") + 1], "[vout]")

    def test_speed_uses_setpts(self):
        command = vd._build_combine_ffmpeg_command(
            [self._plan(speed=2.0)],
            "out.mp4",
            width=64,
            height=96,
            video_fps=30,
            threads=2,
            codec="libx264",
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("setpts=PTS/2.000000", graph)

    def test_max_duration_truncates(self):
        command = vd._build_combine_ffmpeg_command(
            [self._plan()],
            "out.mp4",
            width=64,
            height=96,
            video_fps=30,
            threads=2,
            codec="libx264",
            max_duration=9.5,
        )
        # `-t` also bounds each source read; the output cap is the last one.
        self.assertEqual(command[-4], "-t")
        self.assertEqual(command[-3], "9.500")

    def test_fade_transitions(self):
        cases = [("FadeIn", "fade=t=in:st=0.000:d=1.000"),
                 ("FadeOut", "fade=t=out:st=1.000:d=1.000")]
        for transition, expected in cases:
            with self.subTest(transition=transition):
                command = vd._build_combine_ffmpeg_command(
                    [self._plan(output_duration=2.0, transition=transition)],
                    "out.mp4", width=64, height=96, video_fps=30,
                    threads=2, codec="libx264",
                )
                graph = command[command.index("-filter_complex") + 1]
                self.assertIn(expected, graph)

    def test_slide_and_zoom_transitions(self):
        cases = [
            ("SlideIn", "left", "overlay="),
            ("SlideOut", "bottom", "overlay="),
            ("ZoomIn", None, "zoompan="),
            ("ZoomOut", None, "zoompan="),
        ]
        for transition, side, expected in cases:
            with self.subTest(transition=transition):
                command = vd._build_combine_ffmpeg_command(
                    [
                        self._plan(
                            output_duration=2.0,
                            transition=transition,
                            transition_side=side,
                        )
                    ],
                    "out.mp4", width=64, height=96, video_fps=30,
                    threads=2, codec="libx264",
                )
                graph = command[command.index("-filter_complex") + 1]
                self.assertIn(expected, graph)
                if expected == "overlay=":
                    self.assertIn("color=c=black", graph)

    def test_no_transition_uses_null(self):
        command = vd._build_combine_ffmpeg_command(
            [self._plan(transition=None)],
            "out.mp4", width=64, height=96, video_fps=30,
            threads=2, codec="libx264",
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("[s0]null[v0]", graph)


class TestChooseTransition(unittest.TestCase):
    def test_each_transition_mode(self):
        cases = [
            (VideoTransitionMode.none, None),
            (None, None),
            (VideoTransitionMode.fade_in, "FadeIn"),
            (VideoTransitionMode.fade_out, "FadeOut"),
            (VideoTransitionMode.slide_in, "SlideIn"),
            (VideoTransitionMode.slide_out, "SlideOut"),
            (VideoTransitionMode.zoom_in, "ZoomIn"),
            (VideoTransitionMode.zoom_out, "ZoomOut"),
        ]
        for mode, expected in cases:
            with self.subTest(mode=mode):
                random.seed(1)
                transition, _ = vd._choose_transition(mode)
                self.assertEqual(transition, expected)

    def test_shuffle_resolves_to_a_known_transition(self):
        random.seed(3)
        transition, side = vd._choose_transition(VideoTransitionMode.shuffle)
        self.assertIn(transition, vd._SHUFFLE_TRANSITIONS)
        if transition in vd._SLIDE_TRANSITIONS:
            self.assertIn(side, vd._TRANSITION_SIDES)
        else:
            self.assertIsNone(side)


class TestLegacyPlanner(unittest.TestCase):
    def test_selection_loops_to_cover_audio(self):
        durations = {"a.mp4": 1.0, "b.mp4": 1.0}
        with patch.object(vd, "_probe_video_duration", side_effect=durations.get):
            random.seed(0)
            plans, total = vd._plan_combine_clips_legacy(
                ["a.mp4", "b.mp4"],
                required_video_duration=2.1,
                max_clip_duration=5,
                video_concat_mode=VideoConcatMode.sequential,
                clip_speed=1.0,
                transition_value=None,
            )
        self.assertGreaterEqual(total, 2.1)
        self.assertEqual(len(plans), 3)  # 1s + 1s + looped 1s
        self.assertTrue(all(plan.output_duration == 1.0 for plan in plans))

    def test_speed_scales_source_window_and_output(self):
        durations = {"a.mp4": 10.0}
        with patch.object(vd, "_probe_video_duration", side_effect=durations.get):
            random.seed(0)
            plans, _ = vd._plan_combine_clips_legacy(
                ["a.mp4"],
                required_video_duration=2.0,
                max_clip_duration=3,
                video_concat_mode=VideoConcatMode.sequential,
                clip_speed=2.0,
                transition_value=None,
            )
        first = plans[0]
        self.assertEqual(first.speed, 2.0)
        self.assertAlmostEqual(first.source_end - first.source_start, 6.0)
        self.assertAlmostEqual(first.output_duration, 3.0)

    def test_placeholder_segment_holes_have_no_transition(self):
        segments = [
            {"index": 1, "duration": 2.0, "clips": [], "holes": []},
        ]
        plans, total = vd._plan_combine_clips_segment_first(
            segments,
            max_clip_duration=5,
            clip_speed=1.0,
            transition_value=VideoTransitionMode.fade_in,
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].source_path, "")
        self.assertIsNone(plans[0].transition)  # placeholders bypass transitions
        self.assertAlmostEqual(total, 2.0)

    def test_segment_holes_become_black_placements(self):
        durations = {"a.mp4": 10.0}
        with patch.object(vd, "_probe_video_duration", side_effect=durations.get):
            random.seed(0)
            plans, _ = vd._plan_combine_clips_segment_first(
                [{"index": 1, "duration": 2.0, "clips": ["a.mp4"], "holes": [0]}],
                max_clip_duration=5,
                clip_speed=1.0,
                transition_value=None,
            )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].source_path, "")
        self.assertAlmostEqual(plans[0].output_duration, 2.0)


class TestCombineRendererParity(unittest.TestCase):
    """The two renderers must produce the same timeline for the default mode."""

    def test_rendered_output_matches_moviepy(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.mp4")
            second = os.path.join(tmp, "second.mp4")
            audio = os.path.join(tmp, "voice.wav")
            moviepy_out = os.path.join(tmp, "moviepy.mp4")
            filter_out = os.path.join(tmp, "filter.mp4")
            _make_clip(first, (200, 30, 30), seconds=1.0)
            _make_clip(second, (30, 30, 200), seconds=1.0)
            _silent_wav(audio, seconds=2.0)

            random.seed(42)
            moviepy_result = vd._combine_videos_moviepy(
                combined_video_path=moviepy_out,
                video_paths=[first, second],
                audio_file=audio,
                video_aspect=VideoAspect.portrait,
                video_concat_mode=VideoConcatMode.sequential,
                video_transition_mode=None,
                max_clip_duration=5,
                threads=2,
            )
            random.seed(42)
            filter_result = vd._combine_videos_ffmpeg_filter(
                combined_video_path=filter_out,
                video_paths=[first, second],
                audio_file=audio,
                video_aspect=VideoAspect.portrait,
                video_concat_mode=VideoConcatMode.sequential,
                video_transition_mode=None,
                max_clip_duration=5,
                threads=2,
            )

            # Both renderers share the same contract: return the combined path.
            self.assertEqual(moviepy_result, moviepy_out)
            self.assertEqual(filter_result, filter_out)
            self.assertTrue(os.path.getsize(filter_out) > 0)
            moviepy_clip = VideoFileClip(moviepy_out)
            filter_clip = VideoFileClip(filter_out)
            try:
                self.assertEqual(moviepy_clip.size, filter_clip.size)
                self.assertAlmostEqual(
                    moviepy_clip.duration, filter_clip.duration, delta=0.2
                )
                for t in (0.2, 0.9, 1.5, 1.9):
                    a = moviepy_clip.get_frame(t).astype(np.int16)
                    b = filter_clip.get_frame(t).astype(np.int16)
                    self.assertLess(
                        float(np.mean(np.abs(a - b))),
                        12.0,
                        msg=f"frames diverge at t={t}",
                    )
            finally:
                moviepy_clip.close()
                filter_clip.close()


class TestTransitionParity(unittest.TestCase):
    """Transition-style parity: the fast path must reproduce the MoviePy look."""

    def _render_both(self, tmp, transition):
        first = os.path.join(tmp, "first.mp4")
        audio = os.path.join(tmp, "voice.wav")
        moviepy_out = os.path.join(tmp, "moviepy.mp4")
        filter_out = os.path.join(tmp, "filter.mp4")
        _make_clip(first, (210, 40, 40), seconds=1.5)
        _silent_wav(audio, seconds=1.5)
        for renderer, out in (
            (vd._combine_videos_moviepy, moviepy_out),
            (vd._combine_videos_ffmpeg_filter, filter_out),
        ):
            random.seed(7)
            renderer(
                combined_video_path=out,
                video_paths=[first],
                audio_file=audio,
                video_aspect=VideoAspect.portrait,
                video_concat_mode=VideoConcatMode.sequential,
                video_transition_mode=transition,
                max_clip_duration=5,
                threads=2,
            )
        return moviepy_out, filter_out

    def test_fade_in_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            moviepy_out, filter_out = self._render_both(
                tmp, VideoTransitionMode.fade_in
            )
            a_clip = VideoFileClip(moviepy_out)
            b_clip = VideoFileClip(filter_out)
            try:
                self.assertEqual(a_clip.size, b_clip.size)
                # A fade-in darkens the opening, then the clip brightens.
                first_frame = b_clip.get_frame(0.02).astype(np.int16)
                mid_frame = b_clip.get_frame(0.6).astype(np.int16)
                last_frame = b_clip.get_frame(1.4).astype(np.int16)
                self.assertLess(float(first_frame.mean()), 40.0)
                self.assertLess(float(first_frame.mean()), float(mid_frame.mean()))
                self.assertLess(float(mid_frame.mean()), float(last_frame.mean()))
                for t in (0.4, 0.9, 1.4):
                    delta = float(
                        np.mean(
                            np.abs(
                                a_clip.get_frame(t).astype(np.int16)
                                - b_clip.get_frame(t).astype(np.int16)
                            )
                        )
                    )
                    self.assertLess(delta, 32.0, msg=f"fade diverges at t={t}")
            finally:
                a_clip.close()
                b_clip.close()


class TestPlannerEquivalence(unittest.TestCase):
    """The ffmpeg planner must place clips exactly where MoviePy would."""

    def test_legacy_planner_matches_moviepy_source_ranges(self):
        source_ranges = []

        class _FakeAudioClip:
            duration = 5.9

            def close(self):
                pass

        class _FakeVideoClip:
            def __init__(self, duration, record=False):
                self.duration = duration
                self.size = (1080, 1920)
                self.w = 1080
                self.h = 1920
                self.record = record

            def subclipped(self, start_time, end_time):
                if self.record:
                    source_ranges.append((start_time, end_time))
                return _FakeVideoClip(end_time - start_time)

            def with_speed_scaled(self, factor):
                return _FakeVideoClip(self.duration / factor)

            def close(self):
                pass

        durations = {"a.mp4": 4.0, "b.mp4": 2.0}

        def fake_open(path, audio=False):
            return _FakeVideoClip(durations[path], record=True)

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(vd, "AudioFileClip", return_value=_FakeAudioClip()),
                patch.object(vd, "_open_video_clip_quietly", side_effect=fake_open),
                patch.object(vd, "_write_videofile_with_codec_fallback"),
                patch.object(vd, "_prioritize_unique_source_clips",
                             side_effect=lambda subclipped_items, concat_mode: subclipped_items),
                patch.object(vd, "concat_video_clips_with_ffmpeg"),
            ):
                random.seed(11)
                vd._combine_videos_moviepy(
                    combined_video_path=os.path.join(tmp, "out.mp4"),
                    video_paths=list(durations.keys()),
                    audio_file=os.path.join(tmp, "a.wav"),
                    video_aspect=VideoAspect.portrait,
                    video_concat_mode=VideoConcatMode.sequential,
                    video_transition_mode=None,
                    max_clip_duration=3,
                    clip_speed=1.0,
                )

        random.seed(11)
        with patch.object(vd, "_probe_video_duration", side_effect=durations.get):
            plans, _ = vd._plan_combine_clips_legacy(
                list(durations.keys()),
                required_video_duration=vd._get_required_video_duration(5.9),
                max_clip_duration=3,
                video_concat_mode=VideoConcatMode.sequential,
                clip_speed=1.0,
                transition_value=None,
            )
        plan_ranges = [(p.source_start, p.source_end) for p in plans]
        # The first placements must match exactly; any additional placements
        # are deterministic replays of the same source windows (the MoviePy
        # path replays already-rendered files, so it never re-subclips them).
        self.assertEqual(plan_ranges[: len(source_ranges)], source_ranges)
        self.assertGreaterEqual(len(plans), len(source_ranges))
        for plan in plans[len(source_ranges):]:
            self.assertIn((plan.source_start, plan.source_end), source_ranges)


class TestCodecFallback(unittest.TestCase):
    def test_retries_with_default_codec_and_disables_failing_one(self):
        calls = []

        def build_command(codec):
            calls.append(codec)
            return ["ffmpeg", codec]

        import types

        def fake_run(command, capture_output, text, check):
            if command[-1] == "h264_nvenc":
                return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with (
            patch.object(vd, "_get_effective_video_codec", return_value="h264_nvenc"),
            patch.object(vd, "_disable_runtime_video_codec") as disable,
            patch.object(vd.subprocess, "run", side_effect=fake_run),
        ):
            result = vd._run_ffmpeg_with_codec_fallback(build_command, label="test")

        self.assertEqual(result, "libx264")
        self.assertEqual(calls, ["h264_nvenc", "libx264"])
        disable.assert_called_once()

    def test_does_not_retry_when_already_on_default_codec(self):
        import types

        def build_command(codec):
            return ["ffmpeg", codec]

        def fake_run(command, capture_output, text, check):
            return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with (
            patch.object(vd, "_get_effective_video_codec", return_value="libx264"),
            patch.object(vd.subprocess, "run", side_effect=fake_run),
        ):
            with self.assertRaises(RuntimeError):
                vd._run_ffmpeg_with_codec_fallback(build_command, label="test")


class TestCombineDispatch(unittest.TestCase):
    def test_ffmpeg_filter_mode_dispatches(self):
        from app.config import config

        original = dict(config.app)
        config.app["combine_render_mode"] = "ffmpeg_filter"
        sentinel = object()
        try:
            with patch.dict(
                vd._COMBINE_RENDER_IMPLEMENTATIONS,
                {"ffmpeg_filter": lambda **kwargs: sentinel},
            ):
                result = vd.combine_videos(
                    combined_video_path="c.mp4",
                    video_paths=[],
                    audio_file="a.mp3",
                )
            self.assertIs(result, sentinel)
        finally:
            config.app.clear()
            config.app.update(original)


if __name__ == "__main__":
    unittest.main()
