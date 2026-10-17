"""image_gen 服务单元测试：LLM 精炼降级、Kolors 重试、provider 图片回退、
still→mp4 转换与 make_subject_clip 编排（全 mock，ffmpeg 用真实二进制渲
极短视频）。"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.services import image_gen


def _ffmpeg_cli_available() -> bool:
    try:
        from app.utils import utils

        utils.get_ffmpeg_binary()
    except Exception:
        return False
    return shutil.which("ffprobe") is not None


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


@unittest.skipUnless(_ffmpeg_cli_available(), "ffmpeg/ffprobe binary not available")
class TestStillToClipDimensionNormalization(unittest.TestCase):
    """回归：libx264/yuv420p 要求宽高为偶数。真实下载照片（如 pexels
    原图 3743x5615）常为奇数尺寸，未归一化时 ffmpeg 报 "width not
    divisible by 2" 并留下 0 字节 clip，段回填随之失败。测试只写临时目录。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mpt-still2clip-")
        self.addCleanup(self._tmpdir.cleanup)

    def _make_still(self, name: str, lavfi_source: str) -> str:
        path = os.path.join(self._tmpdir.name, name)
        from app.utils import utils

        subprocess.run(
            [
                utils.get_ffmpeg_binary(),
                "-y",
                "-f",
                "lavfi",
                "-i",
                lavfi_source,
                "-frames:v",
                "1",
                path,
            ],
            check=True,
            capture_output=True,
        )
        return path

    def _probe_video_stream(self, path: str) -> tuple[int, int, float]:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        width, height, duration = [
            token
            for line in result.stdout.splitlines()
            for token in line.split(",")
            if token.strip()
        ]
        return int(width), int(height), float(duration)

    def test_odd_dimension_jpeg_produces_even_sized_clip(self):
        src = self._make_still("odd.jpg", "testsrc=size=375x201:duration=1")
        self.assertEqual(self._probe_video_stream(src)[:2], (375, 201))  # 源头确实两边都是奇数
        clip = image_gen.still_to_clip(src, self._tmpdir.name, duration=0.4)
        self.assertTrue(clip, "奇数尺寸 still 转 clip 失败（0 字节/空产物回归）")
        self.assertTrue(os.path.exists(clip))
        self.assertGreater(os.path.getsize(clip), 0)
        width, height, duration = self._probe_video_stream(clip)
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)
        self.assertGreater(duration, 0)

    def test_even_dimension_still_keeps_exact_dimensions(self):
        src = self._make_still("even.png", "color=c=green:s=320x240:d=0.1")
        clip = image_gen.still_to_clip(src, self._tmpdir.name, duration=0.4)
        self.assertTrue(clip)
        width, height, _ = self._probe_video_stream(clip)
        self.assertEqual((width, height), (320, 240))  # 偶数源不得被意外重采样


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


class TestBackfillFraming(unittest.TestCase):
    def test_backfill_framings_list_order(self):
        expected = [
            "Wide establishing shot",
            "Close-up shot with shallow depth of field",
            "Low-angle shot",
            "Medium shot from behind the subject",
            "Detail shot emphasizing texture and hands",
        ]
        self.assertEqual(image_gen._BACKFILL_FRAMINGS, expected)

    def test_backfill_framing_rotation(self):
        # slot=0, segment_index=0 -> index 0
        self.assertEqual(image_gen.backfill_framing(0), "Wide establishing shot")
        # slot=5, segment_index=1 -> index (5+1)%5=1
        self.assertEqual(
            image_gen.backfill_framing(5, 1),
            "Close-up shot with shallow depth of field",
        )

    def test_backfill_framing_wraps(self):
        self.assertEqual(
            image_gen.backfill_framing(4, 0),
            "Detail shot emphasizing texture and hands",
        )
        self.assertEqual(
            image_gen.backfill_framing(0, 5),
            "Wide establishing shot",
        )


class TestMakeSubjectClipFraming(unittest.TestCase):
    def test_refined_prompt_skips_refine(self):
        """(a) refined_prompt="X", framing="F" -> refine NOT called,
        prompt == "F, X", record["framing"]=="F", record["prompt"]=="F, X"."""
        refine_spy = unittest.mock.MagicMock(return_value="should-not-be-called")
        with (
            patch.object(image_gen, "refine_scene_prompt", refine_spy),
            patch.object(
                image_gen,
                "generate_kolors_image",
                return_value="/materials/gen.png",
            ),
            patch.object(
                image_gen, "still_to_clip", return_value="/materials/gen.mp4"
            ),
        ):
            clip, record = image_gen.make_subject_clip(
                "seg text",
                "subject",
                "9:16",
                "/materials",
                refined_prompt="A pre-refined prompt",
                framing="Close-up shot with shallow depth of field",
            )
        self.assertFalse(refine_spy.called, "refine_scene_prompt should NOT be called")
        self.assertEqual(record["prompt"], "Close-up shot with shallow depth of field, A pre-refined prompt")
        self.assertEqual(record["framing"], "Close-up shot with shallow depth of field")

    def test_refined_prompt_none_calls_refine(self):
        """(b) refined_prompt=None -> refine called with (segment_text, subject_term)."""
        with patch.object(
            image_gen, "refine_scene_prompt", return_value="refined"
        ) as mock_refine, patch.object(
            image_gen,
            "generate_kolors_image",
            return_value="/materials/gen.png",
        ), patch.object(
            image_gen, "still_to_clip", return_value="/materials/gen.mp4"
        ):
            clip, record = image_gen.make_subject_clip(
                "seg text", "subject", "9:16", "/materials", refined_prompt=None
            )
        mock_refine.assert_called_once_with("seg text", "subject")
        self.assertEqual(record["prompt"], "refined")

    def test_refined_prompt_empty_calls_refine(self):
        """(c) refined_prompt="" -> refine called (empty string is not pre-refined)."""
        with patch.object(
            image_gen, "refine_scene_prompt", return_value="refined"
        ) as mock_refine, patch.object(
            image_gen,
            "generate_kolors_image",
            return_value="/materials/gen.png",
        ), patch.object(
            image_gen, "still_to_clip", return_value="/materials/gen.mp4"
        ):
            clip, record = image_gen.make_subject_clip(
                "seg text", "subject", "9:16", "/materials", refined_prompt=""
            )
        mock_refine.assert_called_once()
        self.assertEqual(record["prompt"], "refined")

    def test_framing_prefixes_prompt(self):
        """Framing is prepended to the prompt."""
        with patch.object(
            image_gen, "refine_scene_prompt", return_value="a scene"
        ), patch.object(
            image_gen,
            "generate_kolors_image",
            return_value="/materials/gen.png",
        ), patch.object(
            image_gen, "still_to_clip", return_value="/materials/gen.mp4"
        ):
            _, record = image_gen.make_subject_clip(
                "seg", "subj", "9:16", "/materials", framing="Low-angle shot"
            )
        self.assertEqual(record["framing"], "Low-angle shot")
        self.assertEqual(record["prompt"], "Low-angle shot, a scene")

    def test_record_has_framing_key(self):
        """record always contains 'framing' key."""
        with patch.object(
            image_gen, "refine_scene_prompt", return_value="x"
        ), patch.object(
            image_gen,
            "generate_kolors_image",
            return_value="/materials/gen.png",
        ), patch.object(
            image_gen, "still_to_clip", return_value="/materials/gen.mp4"
        ):
            _, record = image_gen.make_subject_clip(
                "seg", "subj", "9:16", "/materials"
            )
        self.assertIn("framing", record)
        self.assertEqual(record["framing"], "")

    def test_all_sources_fail_framing_still_set(self):
        """Failure path: all sources fail -> record["source"]=="failed",
        record["framing"] still set, returns ("", record)."""
        with patch.object(
            image_gen, "refine_scene_prompt", return_value="x"
        ), patch.object(
            image_gen, "generate_kolors_image", return_value=""
        ), patch.object(
            image_gen, "search_provider_photo", return_value=""
        ):
            clip, record = image_gen.make_subject_clip(
                "seg",
                "subj",
                "9:16",
                "/materials",
                framing="Wide establishing shot",
            )
        self.assertEqual(clip, "")
        self.assertEqual(record["source"], "failed")
        self.assertEqual(record["framing"], "Wide establishing shot")


class TestStillToClipKey(unittest.TestCase):
    def _make_image(self, path):
        import subprocess
        from app.utils import utils as _utils

        subprocess.run(
            [
                _utils.get_ffmpeg_binary(),
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x240:d=0.1",
                "-frames:v",
                "1",
                path,
            ],
            check=True,
            capture_output=True,
        )

    def test_default_key_same_as_current_naming(self):
        """(d) default key="" must reproduce the CURRENT naming."""
        from app.utils import utils as _utils

        img = os.path.join(self._tmp(), "naming_test.png")
        self._make_image(img)
        old_name = f"imgclip-{_utils.md5(img)}.mp4"
        new_name = os.path.basename(
            image_gen.still_to_clip(img, self._tmp(), duration=0.4)
        )
        self.assertEqual(new_name, old_name)

    def test_different_key_produces_different_filename(self):
        """(d) same image, different (framing, duration) -> distinct clip filenames."""
        img = os.path.join(self._tmp(), "key_test.png")
        self._make_image(img)
        clip1 = image_gen.still_to_clip(img, self._tmp(), duration=0.4, key="Wide:3.0")
        clip2 = image_gen.still_to_clip(img, self._tmp(), duration=0.4, key="Close-up:5.0")
        self.assertTrue(clip1)
        self.assertTrue(clip2)
        self.assertNotEqual(os.path.basename(clip1), os.path.basename(clip2))

    def test_same_key_produces_same_filename(self):
        """(d) same image, same (framing, duration) -> same filename."""
        img = os.path.join(self._tmp(), "key_same_test.png")
        self._make_image(img)
        clip1 = image_gen.still_to_clip(img, self._tmp(), duration=0.4, key="Wide:3.0")
        clip2 = image_gen.still_to_clip(img, self._tmp(), duration=0.4, key="Wide:3.0")
        self.assertTrue(clip1)
        self.assertTrue(clip2)
        self.assertEqual(os.path.basename(clip1), os.path.basename(clip2))

    def _tmp(self):
        d = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "test-image-gen")
        os.makedirs(d, exist_ok=True)
        return d


if __name__ == "__main__":
    unittest.main()
