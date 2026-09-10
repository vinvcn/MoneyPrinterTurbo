"""image_gen 服务单元测试：LLM 精炼降级、Kolors 重试、provider 图片回退、
still→mp4 转换与 make_subject_clip 编排（全 mock，ffmpeg 用真实二进制渲
极短视频）。"""

import os
import unittest
from unittest.mock import patch

from app.services import image_gen


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, raise_exc=None):
        self._payload = payload
        self.status_code = status_code
        self._raise_exc = raise_exc
        self.headers = {"Content-Type": "image/jpeg"}

    def json(self):
        if self._raise_exc:
            raise self._raise_exc
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    @property
    def content(self):
        return b"fake-image-bytes"


class TestRefineScenePrompt(unittest.TestCase):
    def test_uses_llm_scene_description(self):
        with patch.object(
            image_gen.llm, "generate_response", return_value="a giant panda cub on snow"
        ):
            self.assertEqual(
                image_gen.refine_scene_prompt("熊猫幼崽在雪地里玩耍"),
                "a giant panda cub on snow",
            )

    def test_falls_back_to_narration_on_llm_error(self):
        with patch.object(
            image_gen.llm, "generate_response", return_value="Error: balance 0"
        ):
            self.assertEqual(
                image_gen.refine_scene_prompt("熊猫幼崽在雪地里玩耍"),
                "熊猫幼崽在雪地里玩耍",
            )

    def test_strips_quotes_and_fallback_on_empty(self):
        with patch.object(image_gen.llm, "generate_response", return_value='  "  "  '):
            self.assertEqual(image_gen.refine_scene_prompt("晨雾中的竹林"), "晨雾中的竹林")


class TestGenerateKolorsImage(unittest.TestCase):
    def test_success_downloads_promptly(self):
        post = _FakeResponse(payload={"images": [{"url": "https://cdn/x.png"}]})
        get = _FakeResponse()
        get.headers = {"Content-Type": "image/png"}
        with (
            patch.object(image_gen.requests, "post", return_value=post) as mock_post,
            patch.object(image_gen.requests, "get", return_value=get) as mock_get,
            patch.object(image_gen.utils, "storage_dir", return_value=self._tmp()),
        ):
            path = image_gen.generate_kolors_image("a panda", "720x1280", self._tmp())
        self.assertTrue(path.endswith(".png"))
        self.assertTrue(os.path.exists(path))
        self.assertEqual(mock_post.call_count, 1)
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "Kwai-Kolors/Kolors")
        self.assertEqual(body["image_size"], "720x1280")
        self.assertEqual(mock_get.call_args.args[0], "https://cdn/x.png")
        self.assertIn("siliconflow", mock_post.call_args.args[0])

    def test_retries_twice_on_5xx_then_succeeds(self):
        responses = [
            _FakeResponse(status_code=503),
            _FakeResponse(status_code=429),
            _FakeResponse(payload={"images": [{"url": "https://cdn/y.png"}]}),
        ]
        with (
            patch.object(image_gen.requests, "post", side_effect=responses),
            patch.object(image_gen.requests, "get", return_value=_FakeResponse()),
            patch.object(image_gen.time, "sleep") as mock_sleep,
            patch.object(image_gen.utils, "storage_dir", return_value=self._tmp()),
        ):
            path = image_gen.generate_kolors_image("a panda", "720x1280", self._tmp())
        self.assertTrue(path)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_returns_empty_after_exhausted_retries(self):
        with (
            patch.object(image_gen.requests, "post", side_effect=RuntimeError("boom")),
            patch.object(image_gen.time, "sleep"),
            patch.object(image_gen.utils, "storage_dir", return_value=self._tmp()),
        ):
            path = image_gen.generate_kolors_image("a panda", "720x1280", self._tmp())
        self.assertEqual(path, "")

    def _tmp(self):
        d = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "test-image-gen")
        os.makedirs(d, exist_ok=True)
        return d


class TestSearchProviderPhoto(unittest.TestCase):
    def test_pexels_photo_downloaded(self):
        pexels = _FakeResponse(
            payload={"photos": [{"src": {"original": "https://img/p.jpg"}}]}
        )
        with (
            patch.object(image_gen.requests, "get", side_effect=[pexels, pexels]) as mock_get,
            patch.object(image_gen.material, "get_api_key", return_value="k"),
            patch.object(image_gen.utils, "storage_dir", return_value=self._tmp()),
        ):
            path = image_gen.search_provider_photo("giant panda", "portrait", self._tmp())
        self.assertTrue(path.endswith(".jpg") or path.endswith(".png"))
        self.assertIn("api.pexels.com", mock_get.call_args_list[0].args[0])

    def test_falls_back_to_pixabay_when_pexels_empty(self):
        pexels = _FakeResponse(payload={"photos": []})
        pixabay = _FakeResponse(payload={"hits": [{"largeImageURL": "https://img/pb.jpg"}]})
        with (
            patch.object(image_gen.requests, "get", side_effect=[pexels, pixabay, pixabay]),
            patch.object(image_gen.material, "get_api_key", return_value="k"),
            patch.object(image_gen.utils, "storage_dir", return_value=self._tmp()),
        ):
            path = image_gen.search_provider_photo("giant panda", "portrait", self._tmp())
        self.assertTrue(path)

    def test_empty_when_both_providers_fail(self):
        with (
            patch.object(
                image_gen.requests, "get", side_effect=RuntimeError("network")
            ),
            patch.object(image_gen.material, "get_api_key", return_value="k"),
            patch.object(image_gen.utils, "storage_dir", return_value=self._tmp()),
        ):
            path = image_gen.search_provider_photo("giant panda", "portrait", self._tmp())
        self.assertEqual(path, "")

    def _tmp(self):
        d = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "test-image-gen")
        os.makedirs(d, exist_ok=True)
        return d


class TestStillToClip(unittest.TestCase):
    def test_renders_mp4_with_ffmpeg(self):
        src = os.path.join(self._tmp(), "still.png")
        with open(src, "wb") as f:
            f.write(b"not-a-real-image")  # ffmpeg 会失败？——不，用真实生成
        # 用 Pillow 不可依赖：直接用 ffmpeg 生成一张纯色图作为输入。
        import subprocess

        from app.utils import utils

        subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=green:s=320x240:d=0.1",
                "-frames:v",
                "1",
                src,
            ],
            check=True,
            capture_output=True,
        )
        clip = image_gen.still_to_clip(src, self._tmp(), duration=0.4)
        self.assertTrue(clip.endswith(".mp4"))
        self.assertTrue(os.path.exists(clip))
        self.assertGreater(os.path.getsize(clip), 0)

    def test_returns_empty_on_bad_image(self):
        src = os.path.join(self._tmp(), "broken.png")
        with open(src, "wb") as f:
            f.write(b"not-an-image")
        clip = image_gen.still_to_clip(src, self._tmp(), duration=0.4)
        self.assertEqual(clip, "")

    def _tmp(self):
        d = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "test-image-gen")
        os.makedirs(d, exist_ok=True)
        return d


class TestMakeSubjectClip(unittest.TestCase):
    def test_kolors_success_returns_clip_and_audit(self):
        with (
            patch.object(
                image_gen, "refine_scene_prompt", return_value="a panda on snow"
            ),
            patch.object(
                image_gen,
                "generate_kolors_image",
                return_value="/materials/gen.png",
            ) as mock_kolors,
            patch.object(
                image_gen, "still_to_clip", return_value="/materials/gen.mp4"
            ),
            patch.object(image_gen.material, "get_api_key", return_value="k"),
        ):
            clip, record = image_gen.make_subject_clip(
                "熊猫在雪地", "giant panda", "9:16", "/materials"
            )
        self.assertEqual(clip, "/materials/gen.mp4")
        self.assertEqual(record["source"], "kolors")
        self.assertEqual(record["prompt"], "a panda on snow")
        self.assertEqual(mock_kolors.call_args.args[1], "720x1280")
        self.assertEqual(mock_kolors.call_args.args[2], "/materials")

    def test_kolors_failure_falls_back_to_provider_photo(self):
        calls = {"kolors": 0}

        def fake_kolors(prompt, image_size, save_dir):
            calls["kolors"] += 1
            return ""

        with (
            patch.object(image_gen, "refine_scene_prompt", return_value="a panda"),
            patch.object(image_gen, "generate_kolors_image", side_effect=fake_kolors),
            patch.object(
                image_gen, "search_provider_photo", return_value="/materials/p.jpg"
            ) as mock_photo,
            patch.object(
                image_gen, "still_to_clip", return_value="/materials/p.mp4"
            ),
        ):
            clip, record = image_gen.make_subject_clip(
                "熊猫在雪地", "giant panda", "16:9", "/materials"
            )
        self.assertEqual(clip, "/materials/p.mp4")
        self.assertEqual(record["source"], "provider_photo")
        self.assertEqual(mock_photo.call_args.args[0], "giant panda")
        # 重试语义在 generate_kolors_image 内部（TestGenerateKolorsImage 已覆盖），
        # 这里只验证编排层对 kolors 源调用一次。
        self.assertEqual(calls["kolors"], 1)

    def test_all_sources_fail_returns_empty_with_record(self):
        with (
            patch.object(image_gen, "refine_scene_prompt", return_value="a panda"),
            patch.object(image_gen, "generate_kolors_image", return_value=""),
            patch.object(image_gen, "search_provider_photo", return_value=""),
        ):
            clip, record = image_gen.make_subject_clip(
                "熊猫在雪地", "", "9:16", "/materials"
            )
        self.assertEqual(clip, "")
        self.assertEqual(record["source"], "failed")
        # subject 词条为空时跳过 provider 图片回退（无可用查询词）。
        self.assertEqual(record["error"], "no_image_source")


if __name__ == "__main__":
    unittest.main()
