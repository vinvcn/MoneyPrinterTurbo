import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect, VideoParams
from app.config import config


class TestVideoAspect(unittest.TestCase):
    def test_to_resolution_known_aspects(self):
        self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
        self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))
        self.assertEqual(VideoAspect.square.to_resolution(), (1080, 1080))

    def test_to_resolution_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            VideoAspect.to_resolution("4:5")


class TestVideoParams(unittest.TestCase):
    def test_rejects_non_positive_generation_counts(self):
        for field_name in ("video_clip_duration", "video_count"):
            for value in (0, -1, None):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValidationError):
                        VideoParams(video_subject="Coffee", **{field_name: value})

    def test_accepts_positive_generation_counts(self):
        params = VideoParams(
            video_subject="Coffee", video_clip_duration=1, video_count=1
        )

        self.assertEqual(params.video_clip_duration, 1)
        self.assertEqual(params.video_count, 1)


class TestVideoSourceDefault(unittest.TestCase):
    """video_source 缺省时读 config.toml [app] video_source；请求值优先。

    回归：schema 默认值曾硬编码为 "pexels"，导致 config.toml 里的
    video_source（如 "library"）对 API 调用完全失效。
    """

    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_default_comes_from_config(self):
        config.app["video_source"] = "library"
        params = VideoParams(video_subject="Coffee")
        self.assertEqual(params.video_source, "library")

    def test_explicit_request_value_wins(self):
        config.app["video_source"] = "library"
        params = VideoParams(video_subject="Coffee", video_source="pixabay")
        self.assertEqual(params.video_source, "pixabay")

    def test_empty_or_none_falls_back_to_config(self):
        config.app["video_source"] = "coverr"
        for value in ("", "   ", None):
            with self.subTest(value=value):
                params = VideoParams(video_subject="Coffee", video_source=value)
                self.assertEqual(params.video_source, "coverr")

    def test_missing_config_falls_back_to_pexels(self):
        config.app.pop("video_source", None)
        params = VideoParams(video_subject="Coffee")
        self.assertEqual(params.video_source, "pexels")

    def test_blank_config_falls_back_to_pexels(self):
        config.app["video_source"] = ""
        params = VideoParams(video_subject="Coffee")
        self.assertEqual(params.video_source, "pexels")


if __name__ == "__main__":
    unittest.main()
