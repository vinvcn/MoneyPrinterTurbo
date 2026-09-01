import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import segment_terms


class TestSegmentTerms(unittest.TestCase):
    def test_build_segment_terms_prompt_follows_generate_terms_shape(self):
        prompt = segment_terms.build_segment_terms_prompt(
            video_subject="黑洞",
            segment_text="黑洞是宇宙中最神秘的天体之一。",
            amount=2,
        )
        # 与 generate_terms 的全局提示词保持同一结构，模型已对此格式稳定。
        self.assertIn("# Role: Video Search Terms Generator", prompt)
        self.assertIn("each search term should consist of 1-3 words", prompt)
        self.assertIn("black hole", prompt.lower())  # subject is appended
        self.assertIn("黑洞是宇宙中最神秘的天体之一。", prompt)
        self.assertIn('"black hole visual topic 1"', prompt)

    def test_extract_terms_for_segments_parses_llm_json(self):
        """批处理模式下，一次调用返回所有分段的词，顺序与分段编号一致。"""
        segments = [
            {"index": 0, "text": "First sentence about cities."},
            {"index": 1, "text": "Second sentence about oceans."},
        ]

        def fake_generate_response(prompt, app_config=None):
            # amount=2 → 每段 2 个词：seg0 两个、seg1 两个，共 4 项。
            self.assertIn("exactly 4 strings", prompt)
            return '["city skyline", "urban night", "ocean waves", "sea water"]'

        with patch.object(
            segment_terms.llm, "_generate_response", side_effect=fake_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="world", amount=2
            )

        # 每段取它的第一个词，并附加主题词（与 generate_terms 约束 2 一致）。
        self.assertEqual(terms, {0: "city skyline world", 1: "ocean waves world"})

    def test_extract_terms_retries_wrong_count_then_succeeds(self):
        """数量不符的响应必须重试；_MAX_RETRIES 次内成功即返回。"""
        segments = [{"index": 0, "text": "A sentence."}]
        calls = {"n": 0}

        def flaky_generate_response(prompt, app_config=None):
            calls["n"] += 1
            if calls["n"] < 2:
                return 'not json at all'  # 无法解析出数组
            return '["valid term"]'

        with patch.object(
            segment_terms.llm, "_generate_response", side_effect=flaky_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="s", amount=1
            )
        self.assertEqual(terms, {0: "valid term s"})
        self.assertEqual(calls["n"], 2)

    def test_extract_terms_all_retries_failed_leaves_segments_unresolved(self):
        """持续失败的分段不进入结果，交给素材层回退链兜底。"""
        segments = [{"index": 0, "text": "A sentence."}]
        with patch.object(
            segment_terms.llm,
            "_generate_response",
            return_value='still not json',  # 永远解析不出数组
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="s", amount=1
            )
        self.assertEqual(terms, {})

    def test_extract_terms_skips_empty_segments(self):
        segments = [
            {"index": 0, "text": ""},
            {"index": 1, "text": "Real."},
        ]
        prompts_seen = []

        def fake_generate_response(prompt, app_config=None):
            prompts_seen.append(prompt)
            return '["term one"]'

        with patch.object(
            segment_terms.llm, "_generate_response", side_effect=fake_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="s", amount=1
            )
        self.assertEqual(list(terms.keys()), [1])
        self.assertEqual(len(prompts_seen), 1)

    def test_batched_extraction_groups_segments_into_few_calls(self):
        """一次 LLM 调用应覆盖多个分段，控制成本与请求次数。"""
        segments = [{"index": i, "text": f"Sentence number {i}."} for i in range(6)]
        calls = []

        def fake_generate_response(prompt, app_config=None):
            calls.append(prompt)
            return json.dumps([f"term-{i}" for i in range(len(segments))])

        with patch.object(
            segment_terms.llm, "_generate_response", side_effect=fake_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="s", amount=1, batch_size=6
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(terms[0], "term-0 s")
        self.assertEqual(terms[5], "term-5 s")


if __name__ == "__main__":
    unittest.main()
