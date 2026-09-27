"""
#312 / option C: the `ffmpeg_overlay` final-video renderer.

Covers overlay-artifact generation, the FFmpeg command builder, mode dispatch,
and a rendered pixel-parity check against the MoviePy final renderer.
"""

import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from moviepy import ColorClip, VideoFileClip
from PIL import Image

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config  # noqa: E402
from app.models.schema import VideoAspect, VideoParams  # noqa: E402
from app.services import video as vd  # noqa: E402

FONT_NAME = "STHeitiMedium.ttc"


def _silent_wav(path, seconds=1.0, sample_rate=44100):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def _write_srt(path, text="字幕测试", start="00:00:00,000", end="00:00:01,000"):
    Path(path).write_text(
        f"1\n{start} --> {end}\n{text}\n\n", encoding="utf-8"
    )


def _make_base_video(path, seconds=1.0, size=(1080, 1920), video_fps=30):
    clip = ColorClip(size=size, color=(20, 90, 160)).with_duration(seconds)
    clip.write_videofile(path, codec="libx264", fps=video_fps, audio=False, logger=None)
    clip.close()


def _overlay(start=0.0, end=1.0, x=10, y=20, size=(6, 8)):
    rgba = np.zeros((size[0], size[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = 255
    return vd._SubtitleOverlay(start=start, end=end, x=x, y=y, rgba=rgba)


class TestOverlayCommandBuilder(unittest.TestCase):
    def test_no_overlays_produces_passthrough_vout(self):
        command = vd._build_final_overlay_command(
            overlay_paths=[],
            overlays=[],
            video_path="base.mp4",
            audio_path="voice.wav",
            bgm_file="",
            output_file="out.mp4",
            video_fps=30,
            threads=2,
            codec="libx264",
            voice_volume=1.0,
            bgm_volume=0.2,
            composed_duration=1.0,
            bgm_loop=False,
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(graph, "[0:v]format=yuv420p[vout];[1:a]volume=1.0[aout]")
        self.assertEqual(command[command.index("-map") + 1], "[vout]")
        self.assertEqual(command[command.index("-r") + 1], "30")
        self.assertEqual(command[-1], "out.mp4")

    def test_overlays_are_chained_with_time_windows(self):
        overlays = [_overlay(start=0.1, end=0.5, x=3, y=4), _overlay(start=0.6, end=0.9)]
        command = vd._build_final_overlay_command(
            overlay_paths=["o1.png", "o2.png"],
            overlays=overlays,
            video_path="base.mp4",
            audio_path="voice.wav",
            bgm_file="",
            output_file="out.mp4",
            video_fps=30,
            threads=2,
            codec="libx264",
            voice_volume=1.0,
            bgm_volume=0.2,
            composed_duration=1.0,
            bgm_loop=False,
        )
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("overlay=x=3:y=4:enable='between(t,0.100,0.500)'", graph)
        self.assertIn("[v0][2:v]overlay=", graph)
        self.assertIn("enable='between(t,0.600,0.900)'", graph)
        self.assertTrue(graph.endswith("[v1]format=yuv420p[vout];[3:a]volume=1.0[aout]"))
        # Both image inputs are looped at the video frame rate.
        self.assertEqual(command.count("-loop"), 2)
        self.assertEqual(
            command[command.index("-framerate") + 1], "30"
        )

    def test_bgm_is_mixed_and_looped(self):
        command = vd._build_final_overlay_command(
            overlay_paths=[],
            overlays=[],
            video_path="base.mp4",
            audio_path="voice.wav",
            bgm_file="bgm.mp3",
            output_file="out.mp4",
            video_fps=30,
            threads=2,
            codec="libx264",
            voice_volume=1.0,
            bgm_volume=0.2,
            composed_duration=10.0,
            bgm_loop=True,
        )
        self.assertIn("-stream_loop", command)
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("volume=0.2,afade=t=out:st=7.000:d=3.000", graph)
        self.assertIn("amix=inputs=2:duration=first", graph)
        self.assertIn("normalize=0", graph)


class TestOverlayArtifacts(unittest.TestCase):
    def test_pngs_are_rgba_and_match_geometry(self):
        overlays = [_overlay(x=5, y=6, size=(4, 7))]
        with tempfile.TemporaryDirectory() as tmp:
            paths = vd._write_subtitle_overlay_pngs(overlays, tmp, "final-1")
            self.assertEqual(len(paths), 1)
            with Image.open(paths[0]) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (7, 4))

    def test_build_subtitle_overlays_from_srt(self):
        with tempfile.TemporaryDirectory() as tmp:
            srt = os.path.join(tmp, "subs.srt")
            _write_srt(srt)
            params = VideoParams(video_subject="t", font_name=FONT_NAME)
            overlays = vd._build_subtitle_overlays(
                srt, params, 1080, 1920, os.path.join(vd.utils.font_dir(), FONT_NAME)
            )
        self.assertEqual(len(overlays), 1)
        overlay = overlays[0]
        self.assertAlmostEqual(overlay.start, 0.0)
        self.assertAlmostEqual(overlay.end, 1.0)
        self.assertGreater(overlay.rgba.shape[0], 0)
        self.assertEqual(overlay.rgba.shape[2], 4)


class TestOverlayDispatch(unittest.TestCase):
    def test_mode_registered_and_dispatches(self):
        self.assertIn("ffmpeg_overlay", vd._FINAL_RENDER_IMPLEMENTATIONS)
        original = dict(config.app)
        config.app["final_render_mode"] = "ffmpeg_overlay"
        sentinel = object()
        try:
            with patch.dict(
                vd._FINAL_RENDER_IMPLEMENTATIONS,
                {"ffmpeg_overlay": lambda **kwargs: sentinel},
            ):
                result = vd.generate_video(
                    video_path="v.mp4",
                    audio_path="a.mp3",
                    subtitle_path="",
                    output_file="o.mp4",
                    params=VideoParams(video_subject="t"),
                )
            self.assertIs(result, sentinel)
        finally:
            config.app.clear()
            config.app.update(original)


class TestFinalRenderParity(unittest.TestCase):
    """Pixel-parity vs the MoviePy final renderer (the promotion gate)."""

    def _render_both(self, tmp):
        base = os.path.join(tmp, "base.mp4")
        audio = os.path.join(tmp, "voice.wav")
        srt = os.path.join(tmp, "subs.srt")
        _make_base_video(base, seconds=1.0)
        _silent_wav(audio, seconds=1.0)
        _write_srt(srt)
        moviepy_out = os.path.join(tmp, "moviepy.mp4")
        overlay_out = os.path.join(tmp, "overlay.mp4")
        params = VideoParams(
            video_subject="t",
            video_aspect=VideoAspect.portrait.value,
            subtitle_enabled=True,
            subtitle_position="bottom",
            font_name=FONT_NAME,
            bgm_type="",
            bgm_volume=0,
        )
        vd._generate_video_moviepy(base, audio, srt, moviepy_out, params)
        vd._generate_video_ffmpeg_overlay(base, audio, srt, overlay_out, params)
        return base, moviepy_out, overlay_out

    def test_rendered_output_matches_moviepy_within_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, moviepy_out, overlay_out = self._render_both(tmp)
            self.assertTrue(os.path.getsize(overlay_out) > 0)
            base_clip = VideoFileClip(base)
            moviepy_clip = VideoFileClip(moviepy_out)
            overlay_clip = VideoFileClip(overlay_out)
            try:
                self.assertEqual(moviepy_clip.size, overlay_clip.size)
                self.assertAlmostEqual(
                    moviepy_clip.duration, overlay_clip.duration, delta=0.2
                )
                for t in (0.1, 0.5, 0.9):
                    a = moviepy_clip.get_frame(t).astype(np.int16)
                    b = overlay_clip.get_frame(t).astype(np.int16)
                    self.assertLess(
                        float(np.mean(np.abs(a - b))),
                        25.0,
                        msg=f"overlay diverges from moviepy at t={t}",
                    )
                # The subtitle must actually be present, not a blank frame:
                # the bottom band differs from the untouched base.
                t = 0.5
                base_frame = base_clip.get_frame(t).astype(np.int16)
                overlay_frame = overlay_clip.get_frame(t).astype(np.int16)
                bottom = slice(int(overlay_clip.h * 0.85), overlay_clip.h)
                self.assertGreater(
                    float(
                        np.mean(
                            np.abs(
                                overlay_frame[bottom] - base_frame[bottom]
                            )
                        )
                    ),
                    1.0,
                )
            finally:
                base_clip.close()
                moviepy_clip.close()
                overlay_clip.close()

    def test_failed_render_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base.mp4")
            audio = os.path.join(tmp, "voice.wav")
            _make_base_video(base, seconds=0.5)
            _silent_wav(audio, seconds=0.5)
            out = os.path.join(tmp, "final.mp4")
            params = VideoParams(video_subject="t", bgm_type="", bgm_volume=0)
            with patch.object(
                vd.subprocess, "run"
            ) as run_mock:
                from types import SimpleNamespace

                run_mock.return_value = SimpleNamespace(
                    returncode=1, stdout="", stderr="boom"
                )
                with self.assertRaises(RuntimeError):
                    vd._generate_video_ffmpeg_overlay(
                        base, audio, "", out, params
                    )
            self.assertFalse(os.path.exists(out))
            self.assertFalse(os.path.exists(out + ".partial.mp4"))


if __name__ == "__main__":
    unittest.main()
