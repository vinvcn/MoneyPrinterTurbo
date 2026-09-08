import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import video_match
from app.services.video_match import SegmentQueries, generate_segment_queries


class TestGenerateSegmentQueries(unittest.TestCase):
    def test_happy_path_returns_terms_in_order_with_both_queries(self):
        """一次调用同时产出三类查询：词条有序、主词在前，两段描述原样提取。"""
        calls = {"n": 0}

        def fake_generate_response(prompt):
            calls["n"] += 1
            self.assertIn('"terms"', prompt)
            self.assertIn('"coarse_query"', prompt)
            self.assertIn('"fine_query"', prompt)
            self.assertIn("A segment about pandas eating bamboo.", prompt)
            return json.dumps(
                {
                    "terms": [
                        "panda eating bamboo",
                        "bamboo forest",
                        "panda cub playing",
                    ],
                    "coarse_query": (
                        "A giant panda sits in a lush green bamboo forest "
                        "munching on bamboo stalks."
                    ),
                    "fine_query": (
                        "Close-up of a giant panda gripping a bamboo stalk "
                        "and stripping the leaves with its teeth."
                    ),
                }
            )

        with patch.object(
            video_match.llm, "generate_response", side_effect=fake_generate_response
        ):
            result = generate_segment_queries(
                "panda", "A segment about pandas eating bamboo."
            )

        self.assertEqual(
            result,
            SegmentQueries(
                terms=[
                    "panda eating bamboo panda",
                    "bamboo forest panda",
                    "panda cub playing panda",
                ],
                coarse_query=(
                    "A giant panda sits in a lush green bamboo forest "
                    "munching on bamboo stalks."
                ),
                fine_query=(
                    "Close-up of a giant panda gripping a bamboo stalk "
                    "and stripping the leaves with its teeth."
                ),
            ),
        )
        # 规格约束：单片段恰好一次 LLM 调用。
        self.assertEqual(calls["n"], 1)

    def test_cjk_term_dropped_and_valid_terms_survive(self):
        """单个词条带 CJK 时只丢该词，其余词条按原序存活。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps(
                {
                    "terms": ["ocean waves", "海浪 coast", "stormy coast"],
                    "coarse_query": "The sea under a stormy sky.",
                    "fine_query": "Waves crashing against a rocky shore.",
                }
            ),
        ):
            result = generate_segment_queries("sea", "A sentence about the sea.")

        self.assertEqual(result.terms, ["ocean waves sea", "stormy coast sea"])
        self.assertEqual(result.coarse_query, "The sea under a stormy sky.")
        self.assertEqual(result.fine_query, "Waves crashing against a rocky shore.")

    def test_terms_only_response_leaves_queries_none(self):
        """模型只回词条字段时，缺失的查询字段解析为 None，词条保持完好。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps({"terms": ["ocean waves", "stormy coast"]}),
        ):
            result = generate_segment_queries("sea", "A sentence.")

        self.assertEqual(result.terms, ["ocean waves sea", "stormy coast sea"])
        self.assertIsNone(result.coarse_query)
        self.assertIsNone(result.fine_query)

    def test_malformed_json_fails_open_after_exact_retry_count(self):
        """持续返回垃圾响应：恰好重试 _MAX_RETRIES 次后 fail-open，绝不抛异常。"""
        calls = {"n": 0}

        def junk_generate_response(prompt):
            calls["n"] += 1
            return "not json at all <|}}} broken"

        with patch.object(
            video_match.llm, "generate_response", side_effect=junk_generate_response
        ):
            result = generate_segment_queries("sea", "A sentence.")

        self.assertEqual(
            result, SegmentQueries(terms=[], coarse_query=None, fine_query=None)
        )
        # 与 segment_terms._MAX_RETRIES 同构：range(1, 2+1) → 最多 2 次调用。
        self.assertEqual(calls["n"], 2)

    def test_llm_error_string_is_retried_then_recovers(self):
        """generate_response 的 'Error: ' 失败串必须重试；第二次成功即返回。"""
        calls = {"n": 0}

        def flaky_generate_response(prompt):
            calls["n"] += 1
            if calls["n"] < 2:
                return "Error: provider unreachable"
            return json.dumps(
                {
                    "terms": ["valid term"],
                    "coarse_query": "A scene.",
                    "fine_query": "A moment.",
                }
            )

        with patch.object(
            video_match.llm, "generate_response", side_effect=flaky_generate_response
        ):
            result = generate_segment_queries("s", "A sentence.")

        self.assertEqual(result.terms, ["valid term s"])
        self.assertEqual(calls["n"], 2)

    def test_subject_appending_follows_existing_cjk_rule(self):
        """英文主题词追加进每个词条；CJK 主题词不追加（既有规则镜像）。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps({"terms": ["term one", "term two"]}),
        ):
            english = generate_segment_queries("world", "A sentence.")
        self.assertEqual(english.terms, ["term one world", "term two world"])

        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps({"terms": ["stellar collapse", "dying star"]}),
        ):
            cjk = generate_segment_queries("黑洞", "A sentence.")
        # 中文主题词不再拼接，词条保持纯英文。
        self.assertEqual(cjk.terms, ["stellar collapse", "dying star"])

    def test_more_than_three_terms_clamped_to_three(self):
        """超过 3 个词条截断为前 3 个，顺序保持模型输出。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps(
                {
                    "terms": ["t1", "t2", "t3", "t4", "t5"],
                    "coarse_query": "Scene.",
                    "fine_query": "Moment.",
                }
            ),
        ):
            result = generate_segment_queries("panda", "A sentence.")

        self.assertEqual(result.terms, ["t1 panda", "t2 panda", "t3 panda"])

    def test_zero_valid_terms_yields_empty_list_but_keeps_queries(self):
        """全部词条被 CJK 过滤时 terms 为空列表；查询字段不受影响（降级由调用方处理）。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps(
                {
                    "terms": ["海浪", "珊瑚礁"],
                    "coarse_query": "The sea.",
                    "fine_query": "Waves.",
                }
            ),
        ):
            result = generate_segment_queries("sea", "A sentence.")

        self.assertEqual(result.terms, [])
        self.assertEqual(result.coarse_query, "The sea.")
        self.assertEqual(result.fine_query, "Waves.")

    def test_markdown_fenced_json_is_parsed(self):
        """模型用 ```json 围栏包裹 JSON（前后带说明文字）时必须正确解析。"""
        payload = json.dumps(
            {
                "terms": ["city skyline", "night traffic"],
                "coarse_query": "A city skyline at dusk.",
                "fine_query": "Cars streaming through downtown streets at night.",
            }
        )
        fenced = f"Here are the queries:\n```json\n{payload}\n```\nHope this helps!"
        with patch.object(video_match.llm, "generate_response", return_value=fenced):
            result = generate_segment_queries("city", "A sentence.")

        self.assertEqual(result.terms, ["city skyline city", "night traffic city"])
        self.assertEqual(result.coarse_query, "A city skyline at dusk.")
        self.assertEqual(
            result.fine_query, "Cars streaming through downtown streets at night."
        )


if __name__ == "__main__":
    unittest.main()
