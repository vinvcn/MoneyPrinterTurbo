import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoParams
from app.services import video as vd


class TestRenderModeResolution(unittest.TestCase):
    """ADR-0013: renderer switches are explicit, exhaustive and fail fast."""

    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_resolution_defaults_and_normalization(self):
        cases = [
            (vd.resolve_final_render_mode, None, "moviepy"),
            (vd.resolve_final_render_mode, "", "moviepy"),
            (vd.resolve_final_render_mode, "moviepy", "moviepy"),
            (vd.resolve_final_render_mode, "  MOVIEPY ", "moviepy"),
            (vd.resolve_combine_render_mode, None, "moviepy"),
            (vd.resolve_combine_render_mode, "", "moviepy"),
            (vd.resolve_combine_render_mode, "MoviePy", "moviepy"),
        ]
        for resolve, configured, expected in cases:
            with self.subTest(resolve=resolve.__name__, configured=configured):
                self.assertEqual(resolve(configured), expected)

    def test_unknown_modes_raise_naming_the_switch(self):
        cases = [
            (vd.resolve_final_render_mode, "ffmpeg_overlay", "final_render_mode"),
            (vd.resolve_combine_render_mode, "ffmpeg_filter", "combine_render_mode"),
        ]
        for resolve, bad_value, switch_name in cases:
            with self.subTest(switch=switch_name):
                with self.assertRaises(ValueError) as ctx:
                    resolve(bad_value)
                message = str(ctx.exception)
                self.assertIn(switch_name, message)
                self.assertIn("moviepy", message)

    def test_validate_render_modes(self):
        cases = [
            ({}, ("moviepy", "moviepy")),
            (
                {"final_render_mode": "moviepy", "combine_render_mode": "moviepy"},
                ("moviepy", "moviepy"),
            ),
            ({"final_render_mode": "ffmpeg_overlay"}, None),
            ({"combine_render_mode": "ffmpeg_filter"}, None),
        ]
        for config_values, expected in cases:
            with self.subTest(config_values=config_values):
                if expected is None:
                    with self.assertRaises(ValueError):
                        vd.validate_render_modes(config_values)
                else:
                    self.assertEqual(
                        vd.validate_render_modes(config_values), expected
                    )

    def test_registry_covers_every_supported_mode(self):
        self.assertEqual(
            set(vd._SUPPORTED_FINAL_RENDER_MODES),
            set(vd._FINAL_RENDER_IMPLEMENTATIONS),
        )
        self.assertEqual(
            set(vd._SUPPORTED_COMBINE_RENDER_MODES),
            set(vd._COMBINE_RENDER_IMPLEMENTATIONS),
        )

    def test_generate_video_rejects_unknown_mode_before_doing_work(self):
        config.app["final_render_mode"] = "bogus"
        with self.assertRaises(ValueError):
            vd.generate_video(
                video_path="v.mp4",
                audio_path="a.mp3",
                subtitle_path="",
                output_file="o.mp4",
                params=VideoParams(video_subject="t"),
            )

    def test_generate_video_dispatches_to_the_selected_implementation(self):
        sentinel = object()
        with patch.dict(
            vd._FINAL_RENDER_IMPLEMENTATIONS,
            {"moviepy": lambda **kwargs: sentinel},
        ):
            result = vd.generate_video(
                video_path="v.mp4",
                audio_path="a.mp3",
                subtitle_path="",
                output_file="o.mp4",
                params=VideoParams(video_subject="t"),
            )
        self.assertIs(result, sentinel)

    def test_failures_name_the_mode(self):
        cases = [
            (
                vd._FINAL_RENDER_IMPLEMENTATIONS,
                vd.generate_video,
                "final_render_mode",
                {
                    "video_path": "v.mp4",
                    "audio_path": "a.mp3",
                    "subtitle_path": "",
                    "output_file": "o.mp4",
                    "params": VideoParams(video_subject="t"),
                },
            ),
            (
                vd._COMBINE_RENDER_IMPLEMENTATIONS,
                vd.combine_videos,
                "combine_render_mode",
                {
                    "combined_video_path": "c.mp4",
                    "video_paths": [],
                    "audio_file": "a.mp3",
                },
            ),
        ]
        for implementations, entrypoint, switch_name, kwargs in cases:
            with self.subTest(switch=switch_name):
                with patch.dict(
                    implementations, {"moviepy": _raise_value_error}
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        entrypoint(**kwargs)
                message = str(ctx.exception)
                self.assertIn(switch_name, message)
                self.assertIn("moviepy", message)

    def test_combine_videos_rejects_unknown_mode_before_doing_work(self):
        config.app["combine_render_mode"] = "bogus"
        with self.assertRaises(ValueError):
            vd.combine_videos(
                combined_video_path="c.mp4",
                video_paths=[],
                audio_file="a.mp3",
            )

    def test_combine_videos_dispatches_to_the_selected_implementation(self):
        sentinel = object()
        with patch.dict(
            vd._COMBINE_RENDER_IMPLEMENTATIONS,
            {"moviepy": lambda **kwargs: sentinel},
        ):
            result = vd.combine_videos(
                combined_video_path="c.mp4",
                video_paths=[],
                audio_file="a.mp3",
            )
        self.assertIs(result, sentinel)


class TestStartupValidation(unittest.TestCase):
    """An invalid switch must abort process startup, not the first task."""

    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        # Reload once more with the restored config so the module is healthy
        # for the rest of the test session.
        importlib.reload(self._controller_module())

    @staticmethod
    def _controller_module():
        from app.controllers.v1 import video as controller

        return controller

    def test_controller_import_aborts_on_unknown_mode(self):
        controller = self._controller_module()
        config.app["final_render_mode"] = "bogus"
        with self.assertRaises(ValueError):
            importlib.reload(controller)


def _raise_value_error(**kwargs):
    raise ValueError("boom")


if __name__ == "__main__":
    unittest.main()
