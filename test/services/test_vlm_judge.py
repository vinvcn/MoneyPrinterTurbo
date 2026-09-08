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


class _RecordingLogger:
    """loguru 不走 stdlib logging 树，assertLogs 看不到——用桩记录 INFO 行。"""

    def __init__(self):
        self.infos = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        pass


class _FakeGate:
    """可编程假门：按需返回预设重复记录、None（放行）或直接抛异常。

    签名严格对齐 EmbeddingGate（无 skip_coarse）：judge_candidate 若再
    透传已删除的 kwarg，这里直接 TypeError，调用契约由本测试锁定。
    """

    def __init__(self, record=None, error=None):
        self.record = record
        self.error = error
        self.calls = []

    def judge_candidate_embedding(self, url, data_uri, term=""):
        self.calls.append((url, data_uri, term))
        if self.error is not None:
            raise self.error
        return self.record


_DUP_RECORD = {
    "verdict": "duplicate",
    "reason": "cos=0.950 >= threshold",
    "image_source": "embedding",
    "duplicate_of": "https://accepted.example/video-a",
    "cos": 0.95,
}


class TestJudgeCandidateEmbeddingGate(_VlmConfigMixin, unittest.TestCase):
    """查重门接线（finding G）：gate 在图像就绪后、VLM 之前运行。"""

    def _item(self, url="https://candidate.example/video-b"):
        return SimpleNamespace(
            url=url,
            source_info={
                "asset_id": "vid-abc",
                "thumbnail_url": "https://img.example/t.jpg",
                "page": 2,
            },
        )

    def _run(self, gate=None):
        item = self._item()
        judge_image = patch.object(
            vlm_judge,
            "judge_image",
            return_value=("relevant", "matches", 1),
        )
        thumbnail = patch.object(
            vlm_judge,
            "download_thumbnail_bytes",
            return_value=(b"img-bytes", (800, 420)),
        )
        with thumbnail, judge_image as mocked_judge:
            judge = vlm_judge.make_default_judge(embedding_gate=gate)
            record = judge(
                item=item,
                segment_text="text",
                search_term="black hole",
            )
        return record, mocked_judge

    def test_duplicate_record_returned_without_vlm_call(self):
        gate = _FakeGate(record=dict(_DUP_RECORD))
        recording = _RecordingLogger()
        with patch.object(vlm_judge, "logger", recording):
            record, mocked_judge = self._run(gate)
        self.assertEqual(mocked_judge.call_count, 0)
        self.assertEqual(record["verdict"], "duplicate")
        self.assertEqual(record["term"], "black hole")
        self.assertEqual(record["asset_id"], "vid-abc")
        self.assertEqual(record["reason"], "cos=0.950 >= threshold")
        self.assertEqual(record["image_source"], "embedding")
        self.assertEqual(record["attempts"], 0)
        self.assertEqual(record["page"], 2)
        self.assertEqual(record["duplicate_of"], _DUP_RECORD["duplicate_of"])
        self.assertEqual(record["cos"], 0.95)
        # duplicate 日志逐字节锁定：审计口径与旧版完全一致
        # （T5 校准按该行解析 run log，格式即契约）。
        self.assertIn(
            "embedding gate flagged duplicate: "
            "asset_id=vid-abc, term='black hole', "
            "duplicate_of=https://accepted.example/video-a, cos=0.95",
            recording.infos,
        )

    def test_gate_receives_url_data_uri_and_term(self):
        gate = _FakeGate(record=dict(_DUP_RECORD))
        self._run(gate)
        url, data_uri, term = gate.calls[0]
        self.assertEqual(url, "https://candidate.example/video-b")
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))
        self.assertEqual(term, "black hole")
        self.assertEqual(len(gate.calls), 1)

    def test_gate_pass_through_calls_vlm(self):
        gate = _FakeGate(record=None)
        record, mocked_judge = self._run(gate)
        self.assertEqual(mocked_judge.call_count, 1)
        self.assertEqual(record["verdict"], "relevant")
        self.assertNotIn("duplicate_of", record)
        self.assertNotIn("cos", record)
        self.assertEqual(record["attempts"], 1)

    def test_gate_exception_fails_open_to_vlm(self):
        """gate 实现意外抛异常时照常走 VLM，绝不穿透 judge_candidate。"""
        gate = _FakeGate(error=RuntimeError("boom"))
        record, mocked_judge = self._run(gate)
        self.assertEqual(mocked_judge.call_count, 1)
        self.assertEqual(record["verdict"], "relevant")

    def test_gate_off_matches_head_flow(self):
        """embedding_gate=None（默认）时不触发查重，记录为 HEAD 原有形态。"""
        record, mocked_judge = self._run(gate=None)
        self.assertEqual(mocked_judge.call_count, 1)
        self.assertEqual(record["verdict"], "relevant")
        self.assertEqual(
            sorted(record.keys()),
            ["asset_id", "attempts", "image_source", "page", "reason",
             "term", "verdict"],
        )


if __name__ == "__main__":
    unittest.main()
