import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import image_embedding, video_match
from app.services.image_embedding import EmbeddingGate
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


def _vec_with_cos(cos: float) -> list[float]:
    """以查询向量 [1,0,0] 为基准、余弦恰为 cos 的单位向量。"""
    return [cos, 0.0, math.sqrt(1.0 - cos * cos)]


def _dup_vec() -> list[float]:
    """同时贴近查询 [1,0,0] 与锚点 [0,1,0]（两侧余弦约 0.70）的向量。"""
    return [0.70, 0.70, math.sqrt(1.0 - 0.49 - 0.49)]


class _StubEmbedImage:
    """按 data_uri 查表的受控向量源，记录每次调用供次数断言。"""

    def __init__(self, vectors):
        self.vectors = dict(vectors)
        self.calls = []

    def __call__(self, data_uri, model, api_key, base_url=None, timeout=30.0):
        self.calls.append(data_uri)
        return self.vectors.get(data_uri)


def _candidate(i: int) -> dict:
    return {"asset_id": f"a{i}", "url": f"https://x/{i}", "data_uri": f"data:{i}"}


class TestCoarseRank(unittest.TestCase):
    """粗排：余弦降序 + 查重门走查 + fail-open 三条硬契约。"""

    def _rank(self, pool, cache, gate, vectors, coarse_query="scene"):
        stub = _StubEmbedImage(vectors)
        with (
            patch.object(
                image_embedding, "embed_text", return_value=[1.0, 0.0, 0.0]
            ),
            patch.object(image_embedding, "embed_image", stub),
        ):
            selected, dup_skips = video_match.coarse_rank(
                pool, coarse_query, cache, gate
            )
        return selected, dup_skips, stub

    def test_descending_cosine_order_is_stable(self):
        pool = [_candidate(i) for i in range(6)]
        vectors = {
            "data:0": _vec_with_cos(0.9),
            "data:1": _vec_with_cos(0.5),
            "data:2": _vec_with_cos(0.7),
            "data:3": _vec_with_cos(0.3),
            "data:4": _vec_with_cos(0.95),
            # 与 data:2 同分：稳定性要求平票保持 pool 顺序。
            "data:5": _vec_with_cos(0.7),
        }
        cache: dict[str, list[float]] = {}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=cache
        )
        selected, dup_skips, stub = self._rank(pool, cache, gate, vectors)
        self.assertEqual(
            [c["url"] for c in selected],
            [
                "https://x/4",
                "https://x/0",
                "https://x/2",
                "https://x/5",
                "https://x/1",
                "https://x/3",
            ],
        )
        self.assertEqual(dup_skips, [])
        # 每个 URL 恰好嵌入一次：预热后门走查零重复嵌入。
        self.assertEqual(sorted(stub.calls), sorted(vectors.keys()))

    def test_dup_gate_walk_skips_duplicates_and_refills_to_30(self):
        """5 个重复候选排在最前：走查跳过它们，从次优补足 30 个非重复。"""
        dup_positions = {2, 7, 12, 17, 22}
        pool = []
        vectors = {"data:anchor": [0.0, 1.0, 0.0]}
        n_ok = 0
        n_dup = 0
        for pos in range(35):
            if pos in dup_positions:
                pool.append(
                    {
                        "asset_id": f"d{n_dup}",
                        "url": f"https://dup/{n_dup}",
                        "data_uri": f"data:dup{n_dup}",
                    }
                )
                vectors[f"data:dup{n_dup}"] = _dup_vec()
                n_dup += 1
            else:
                pool.append(
                    {
                        "asset_id": f"n{n_ok}",
                        "url": f"https://ok/{n_ok}",
                        "data_uri": f"data:ok{n_ok}",
                    }
                )
                vectors[f"data:ok{n_ok}"] = _vec_with_cos(0.40 + 0.01 * n_ok)
                n_ok += 1
        stub = _StubEmbedImage(vectors)
        cache: dict[str, list[float]] = {}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=cache
        )
        with (
            patch.object(
                image_embedding, "embed_text", return_value=[1.0, 0.0, 0.0]
            ),
            patch.object(image_embedding, "embed_image", stub),
        ):
            # 公开流程播种锚点：先判定再采纳，锚点向量进入注册表。
            gate.judge_candidate_embedding("https://anchor", "data:anchor")
            gate.register_accepted("https://anchor")
            selected, dup_skips = video_match.coarse_rank(
                pool, "scene", cache, gate
            )

        self.assertEqual(len(selected), 30)
        self.assertEqual(len(dup_skips), 5)
        self.assertEqual(
            [r["url"] for r in dup_skips],
            [f"https://dup/{i}" for i in range(5)],
        )
        self.assertEqual(
            {c["url"] for c in selected}, {f"https://ok/{i}" for i in range(30)}
        )
        # 降序：非重复候选按余弦从高到低（0.69 → 0.40）。
        self.assertEqual(
            [c["url"] for c in selected],
            [f"https://ok/{29 - i}" for i in range(30)],
        )
        # 每 URL（含重复候选与锚点）恰好嵌入一次：走查零新增嵌入。
        self.assertEqual(len(stub.calls), 36)
        self.assertEqual(len(set(stub.calls)), 36)
        for record in dup_skips:
            self.assertEqual(
                set(record.keys()), {"asset_id", "url", "reason"}
            )

    def test_embed_failed_candidate_sinks_without_crashing(self):
        """嵌入失败候选得 -1.0 沉底：排在零分候选之后，不崩溃、不剔除。"""
        pool = [
            {"asset_id": "a", "url": "https://x/a", "data_uri": "data:a"},
            {"asset_id": "b", "url": "https://x/b", "data_uri": "data:b"},
            {"asset_id": "z", "url": "https://x/z", "data_uri": "data:z"},
            {"asset_id": "f", "url": "https://x/f", "data_uri": "data:fail"},
        ]
        vectors = {
            "data:a": _vec_with_cos(0.9),
            "data:b": _vec_with_cos(0.5),
            "data:z": [0.0, 0.0],
        }
        cache: dict[str, list[float]] = {}
        gate = EmbeddingGate(model="m", api_key="k", threshold=0.68)
        selected, dup_skips, _ = self._rank(pool, cache, gate, vectors)
        self.assertEqual(
            [c["url"] for c in selected],
            ["https://x/a", "https://x/b", "https://x/z", "https://x/f"],
        )
        self.assertEqual(dup_skips, [])
        # 失败向量不写缓存；其余三个成功向量在缓存中。
        self.assertNotIn("https://x/f", cache)
        self.assertEqual(len(cache), 3)

    def test_embed_text_none_fails_open_to_pool_slice(self):
        """查询向量不可得：返回 pool[:30]（interleave 路径）+ fail-open 告警。"""
        pool = [_candidate(i) for i in range(35)]
        with (
            patch.object(image_embedding, "embed_text", return_value=None),
            patch.object(image_embedding, "embed_image") as embed_mock,
                    patch.object(video_match, "logger") as mock_logger,
        ):
            selected, dup_skips = video_match.coarse_rank(pool, "scene", {}, None)
        self.assertEqual(
            [c["url"] for c in selected],
            [f"https://x/{i}" for i in range(30)],
        )
        self.assertEqual(dup_skips, [])
        embed_mock.assert_not_called()
        warnings = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("video match: coarse rank failed, fail-open" in w for w in warnings)
        )
        self.assertTrue(any("unavailable" in w for w in warnings))

    def test_missing_coarse_query_fails_open_without_embed_text_call(self):
        """coarse_query 为 None/空串：不走 embed_text，直接 pool[:30] 降级。"""
        pool = [_candidate(i) for i in range(35)]
        for missing in (None, ""):
            with self.subTest(missing=missing):
                with (
                    patch.object(image_embedding, "embed_text") as text_mock,
            patch.object(video_match, "logger") as mock_logger,
                ):
                    selected, dup_skips = video_match.coarse_rank(
                        pool, missing, {}, None
                    )
                self.assertEqual(
                    [c["url"] for c in selected],
                    [f"https://x/{i}" for i in range(30)],
                )
                self.assertEqual(dup_skips, [])
                text_mock.assert_not_called()
                warnings = [
                    str(c.args[0]) for c in mock_logger.warning.call_args_list
                ]
                self.assertTrue(
                    any(
                        "video match: coarse rank failed, fail-open" in w
                        for w in warnings
                    )
                )
                self.assertTrue(any("missing" in w for w in warnings))

    def test_gate_none_skips_duplicate_walk_but_keeps_order(self):
        """查重门未注入：跳过走查直接按排序放行，不产生重复记录。"""
        pool = [
            {"asset_id": "a", "url": "https://x/a", "data_uri": "data:a"},
            {"asset_id": "b", "url": "https://x/b", "data_uri": "data:b"},
        ]
        vectors = {"data:a": _vec_with_cos(0.3), "data:b": _vec_with_cos(0.8)}
        selected, dup_skips, _ = self._rank(pool, {}, None, vectors)
        self.assertEqual(
            [c["url"] for c in selected], ["https://x/b", "https://x/a"]
        )
        self.assertEqual(dup_skips, [])

    def test_vector_cache_none_is_tolerated(self):
        """vector_cache=None：粗排退化为内部临时缓存，照常排序。"""
        pool = [_candidate(i) for i in range(2)]
        vectors = {"data:0": _vec_with_cos(0.6), "data:1": _vec_with_cos(0.8)}
        selected, dup_skips, _ = self._rank(pool, None, None, vectors)
        self.assertEqual([c["url"] for c in selected], ["https://x/1", "https://x/0"])
        self.assertEqual(dup_skips, [])


if __name__ == "__main__":
    unittest.main()
