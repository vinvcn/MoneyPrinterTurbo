"""
VLM 下载前相关性过滤的单元测试（issue #9）。

全部 mock，不发真实网络请求，不依赖真实 VLM 端点。
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import vlm_judge


def _vlm_response(verdict="relevant", reason="matches"):
    return SimpleNamespace(
        status_code=200,
        json=lambda: {
            "choices": [
                {
                    "message": {
                        "content": f'{{"verdict": "{verdict}", "reason": "{reason}"}}'
                    }
                }
            ]
        },
    )


class _VlmConfigMixin:
    def setUp(self):
        self._original_vlm = dict(config.vlm)

    def tearDown(self):
        config.vlm.clear()
        config.vlm.update(self._original_vlm)


class TestJudgeImage(_VlmConfigMixin, unittest.TestCase):
    def test_judge_image_returns_parsed_verdict(self):
        with patch.object(vlm_judge.requests, "post", return_value=_vlm_response()):
            verdict, reason, attempts = vlm_judge.judge_image(
                image_data_uri="data:image/jpeg;base64,AAAA",
                search_term="black hole",
                segment_text="A black hole forms.",
                api_key="test-key",
            )
        self.assertEqual(verdict, "relevant")
        self.assertEqual(reason, "matches")
        self.assertEqual(attempts, 1)

    def test_judge_image_fails_open_after_retries(self):
        """VLM 持续失败时返回 uncertain（fail-open），不抛异常。"""
        error = SimpleNamespace(status_code=500, json=lambda: {})
        with patch.object(vlm_judge.requests, "post", return_value=error):
            verdict, reason, attempts = vlm_judge.judge_image(
                image_data_uri="data:image/jpeg;base64,AAAA",
                search_term="black hole",
                segment_text="text",
                api_key="test-key",
            )
        self.assertEqual(verdict, vlm_judge.VERDICT_UNCERTAIN)
        self.assertEqual(attempts, vlm_judge.JUDGE_MAX_RETRIES)
        self.assertIn("unavailable", reason)

    def test_judge_image_retries_on_unparseable_response(self):
        bad = SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "not json"}}]},
        )
        with patch.object(
            vlm_judge.requests, "post", side_effect=[bad, bad, _vlm_response()]
        ) as post:
            verdict, _, attempts = vlm_judge.judge_image(
                image_data_uri="data:image/jpeg;base64,AAAA",
                search_term="t",
                segment_text="text",
                api_key="k",
            )
        self.assertEqual(post.call_count, 3)
        self.assertEqual(verdict, "relevant")
        self.assertEqual(attempts, 3)

    def test_parse_verdict_rejects_unknown_verdict(self):
        self.assertIsNone(vlm_judge._parse_verdict('{"verdict": "maybe"}'))
        self.assertIsNone(vlm_judge._parse_verdict("[]"))
        self.assertIsNone(vlm_judge._parse_verdict(""))


class TestConfigGating(_VlmConfigMixin, unittest.TestCase):
    def test_disabled_by_default_without_section(self):
        config.vlm.clear()
        self.assertFalse(vlm_judge.is_enabled())

    def test_enabled_flag_respected(self):
        config.vlm["enabled"] = True
        self.assertTrue(vlm_judge.is_enabled())
        config.vlm["enabled"] = False
        self.assertFalse(vlm_judge.is_enabled())

    def test_defaults_for_base_url_and_model(self):
        config.vlm.clear()
        judge_config = vlm_judge.load_judge_config()
        self.assertEqual(judge_config["base_url"], vlm_judge.DEFAULT_BASE_URL)
        self.assertEqual(judge_config["model"], vlm_judge.DEFAULT_MODEL)

    def test_api_key_never_enters_judge_config(self):
        config.vlm["api_key"] = "secret-key"
        self.assertNotIn("secret-key", str(vlm_judge.load_judge_config()))


class TestProbeImageSize(unittest.TestCase):
    def test_png_size(self):
        payload = (
            b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
            + (800).to_bytes(4, "big") + (420).to_bytes(4, "big")
        )
        self.assertEqual(vlm_judge._probe_image_size(payload), (800, 420))

    def test_unknown_format_returns_zero(self):
        self.assertEqual(vlm_judge._probe_image_size(b"RIFFxxxxWEBP"), (0, 0))
        self.assertEqual(vlm_judge._probe_image_size(b""), (0, 0))


if __name__ == "__main__":
    unittest.main()
