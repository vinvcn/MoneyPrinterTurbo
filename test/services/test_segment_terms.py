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
        self.assertIn("translate the main subject of the video into English", prompt)
        self.assertIn("黑洞是宇宙中最神秘的天体之一。", prompt)
        self.assertIn('"black hole visual topic 1"', prompt)
        # 搜索 API 仅接受英文：提示词必须明确 CJK 一律翻译。
        self.assertIn("pure English", prompt)

    def test_batch_prompt_requires_diverse_terms_and_exact_count(self):
        """批处理提示词必须带多样性约束（近似词命中同一素材候选池）。"""
        prompt = segment_terms._build_batch_prompt(
            video_subject="space",
            numbered_segments=[(1, "First segment."), (2, "Second segment.")],
            per_segment=3,
        )
        # 精确数量与顺序约束保持不变。
        self.assertIn("exactly 6 strings", prompt)
        self.assertIn("same order as the numbered segments", prompt)
        # 纯英文约束保持不变。
        self.assertIn("pure English", prompt)
        # 多样性约束：同段词条必须覆盖不同画面角度，禁止同义改写。
        self.assertIn("DIFFERENT visual angles, scenes, or shot types", prompt)
        self.assertIn("never synonyms or minor rephrasings", prompt)

    def test_default_terms_per_segment_is_three(self):
        self.assertEqual(segment_terms.TERMS_PER_SEGMENT, 3)

    def test_extract_terms_returns_ordered_list_per_segment(self):
        """返回 {index: [主词, 备用词, ...]}，词条顺序与模型输出一致。"""
        segments = [{"index": 0, "text": "A sentence about the sea."}]

        with patch.object(
            segment_terms.llm,
            "generate_response",
            return_value='["ocean waves", "underwater reef", "stormy coast"]',
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="sea", amount=3
            )

        self.assertEqual(
            terms,
            {0: ["ocean waves sea", "underwater reef sea", "stormy coast sea"]},
        )

    def test_extract_terms_discards_only_cjk_terms(self):
        """三个词中只有一个带 CJK 时，其余两个按原序存活，分段不被丢弃。"""
        segments = [{"index": 0, "text": "A sentence about the sea."}]

        with patch.object(
            segment_terms.llm,
            "generate_response",
            return_value='["ocean waves", "underwater 人工 reef", "stormy coast"]',
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="sea", amount=3
            )

        self.assertEqual(terms, {0: ["ocean waves sea", "stormy coast sea"]})

    def test_extract_terms_omits_segment_when_all_terms_cjk(self):
        """全词条带 CJK（主题词泄漏）时分段不进入结果，交给素材层回退链。"""
        segments = [{"index": 0, "text": "A sentence about the sea."}]

        with patch.object(
            segment_terms.llm,
            "generate_response",
            return_value='["ocean 海浪", "reef 珊瑚", "coast 海岸"]',
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="sea", amount=3
            )

        self.assertEqual(terms, {})

    def test_cjk_subject_is_not_appended_to_any_term(self):
        """中文主题词不能追加进任何搜索词（UAT: 'stellar collapse 黑洞' 全军覆没）。"""
        segments = [{"index": 0, "text": "A sentence."}]

        with patch.object(
            segment_terms.llm,
            "generate_response",
            return_value='["stellar collapse", "dying star"]',
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="黑洞", amount=2
            )

        # 中文主题词不再拼接，所有词条保持纯英文。
        self.assertEqual(terms, {0: ["stellar collapse", "dying star"]})

    def test_subject_appended_to_each_term(self):
        """英文主题词必须追加到每个词条（与 generate_terms 约束 2 一致）。"""
        segments = [{"index": 0, "text": "A sentence."}]

        with patch.object(
            segment_terms.llm,
            "generate_response",
            return_value='["term one", "term two", "term three"]',
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="world", amount=3
            )

        self.assertEqual(
            terms, {0: ["term one world", "term two world", "term three world"]}
        )

    def test_extract_terms_for_segments_parses_llm_json(self):
        """批处理模式下，每段取自己 per_segment 大小的切片，顺序与编号一致。"""
        segments = [
            {"index": 0, "text": "First sentence about cities."},
            {"index": 1, "text": "Second sentence about oceans."},
        ]

        def fake_generate_response(prompt):
            # 默认 amount=3 → 每段 3 个词：seg0 三个、seg1 三个，共 6 项。
            self.assertIn("exactly 6 strings", prompt)
            return json.dumps(
                [
                    "city skyline",
                    "urban night",
                    "downtown traffic",
                    "ocean waves",
                    "sea water",
                    "deep reef",
                ]
            )

        with patch.object(
            segment_terms.llm, "generate_response", side_effect=fake_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="world"
            )

        # 每段拿到自己的 3 词切片，主词在前，并附加主题词。
        self.assertEqual(
            terms,
            {
                0: ["city skyline world", "urban night world", "downtown traffic world"],
                1: ["ocean waves world", "sea water world", "deep reef world"],
            },
        )

    def test_extract_terms_retries_wrong_count_then_succeeds(self):
        """数量不符的响应必须重试；_MAX_RETRIES 次内成功即返回。"""
        segments = [{"index": 0, "text": "A sentence."}]
        calls = {"n": 0}

        def flaky_generate_response(prompt):
            calls["n"] += 1
            if calls["n"] < 2:
                return 'not json at all'  # 无法解析出数组
            return '["valid term"]'

        with patch.object(
            segment_terms.llm, "generate_response", side_effect=flaky_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="s", amount=1
            )
        self.assertEqual(terms, {0: ["valid term s"]})
        self.assertEqual(calls["n"], 2)

    def test_extract_terms_all_retries_failed_leaves_segments_unresolved(self):
        """持续失败的分段不进入结果，交给素材层回退链兜底。"""
        segments = [{"index": 0, "text": "A sentence."}]
        with patch.object(
            segment_terms.llm,
            "generate_response",
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

        def fake_generate_response(prompt):
            prompts_seen.append(prompt)
            return '["term one"]'

        with patch.object(
            segment_terms.llm, "generate_response", side_effect=fake_generate_response
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

        def fake_generate_response(prompt):
            calls.append(prompt)
            # 默认 amount=3、batch_size=6 → 一次调用返回 18 个词。
            return json.dumps([f"term-{i}" for i in range(len(segments) * 3)])

        with patch.object(
            segment_terms.llm, "generate_response", side_effect=fake_generate_response
        ):
            terms = segment_terms.extract_terms_for_segments(
                segments, video_subject="s", batch_size=6
            )
        self.assertEqual(len(calls), 1)
        # 每段拿到自己 3 词切片：seg i → term-(3i)..term-(3i+2)。
        self.assertEqual(terms[0], ["term-0 s", "term-1 s", "term-2 s"])
        self.assertEqual(terms[5], ["term-15 s", "term-16 s", "term-17 s"])


if __name__ == "__main__":
    unittest.main()
