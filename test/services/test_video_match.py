import json
import math
import random
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo
from app.services import image_embedding, image_gen, video_match
from app.services.image_embedding import EmbeddingGate
from app.services.video import segment_window_plan
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
                    "panda eating bamboo",
                    "bamboo forest",
                    "panda cub playing",
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

    def test_cjk_term_stripped_and_valid_terms_survive(self):
        """混排词条剔除 CJK 保留英文部分；纯 CJK 词条剔后为空直接丢弃。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps(
                {
                    "terms": ["ocean waves", "海浪 coast", "海啸"],
                    "coarse_query": "The sea under a stormy sky.",
                    "fine_query": "Waves crashing against a rocky shore.",
                }
            ),
        ):
            result = generate_segment_queries("sea", "A sentence about the sea.")

        # "海啸" 剔除后为空 → 丢弃（无主题锚点兜底）；词条不带主题后缀。
        self.assertEqual(result.terms, ["ocean waves", "coast"])
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

        self.assertEqual(result.terms, ["ocean waves", "stormy coast"])
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
        # 解析失败重试语义：range(1, 2+1) → 最多 2 次调用。
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

        self.assertEqual(result.terms, ["valid term"])
        self.assertEqual(calls["n"], 2)

    def test_terms_are_never_subject_suffixed(self):
        """词条永不追加主题词：英文主题与 CJK 主题都不拼接（双重追加隐患
        已移除，主题只经 prompt Context 影响词条）。"""
        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps({"terms": ["ocean waves", "stormy coast"]}),
        ):
            english = generate_segment_queries("sea", "A sentence.")
        self.assertEqual(english.terms, ["ocean waves", "stormy coast"])

        with patch.object(
            video_match.llm,
            "generate_response",
            return_value=json.dumps({"terms": ["ocean waves", "stormy coast"]}),
        ):
            cjk = generate_segment_queries("海洋", "A sentence.")
        self.assertEqual(cjk.terms, ["ocean waves", "stormy coast"])

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

        self.assertEqual(result.terms, ["t1", "t2", "t3"])

    def test_cjk_terms_stripped_to_empty(self):
        """纯 CJK 词条剔除后为空 → 全部丢弃，terms=[]（不再降级为主题
        锚点词）；英文与 CJK 主题行为一致。"""
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
            cjk_result = generate_segment_queries("海洋", "A sentence.")

        self.assertEqual(cjk_result.terms, [])
        self.assertEqual(cjk_result.coarse_query, "The sea.")
        self.assertEqual(cjk_result.fine_query, "Waves.")

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

        self.assertEqual(result.terms, ["city skyline", "night traffic"])
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

    def test_empty_data_uri_sinks_without_embed_call(self):
        """空 data_uri（缩略图缺失）直接沉底：不发起必然 400 的嵌入调用，
        查重门走查同样零调用放行（门内空 URI fail-open）。"""
        pool = [
            {"asset_id": "ok", "url": "https://x/ok", "data_uri": "data:ok"},
            {"asset_id": "empty", "url": "https://x/empty", "data_uri": ""},
            {"asset_id": "missing", "url": "https://x/missing"},
        ]
        vectors = {"data:ok": _vec_with_cos(0.9)}
        cache: dict[str, list[float]] = {}
        gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.68, vector_cache=cache
        )
        selected, dup_skips, stub = self._rank(pool, cache, gate, vectors)
        self.assertEqual(
            [c["url"] for c in selected],
            ["https://x/ok", "https://x/empty", "https://x/missing"],
        )
        self.assertEqual(dup_skips, [])
        # 空 URI 候选全程零嵌入调用；成功候选恰好一次（粗排与门共享缓存）。
        self.assertEqual(stub.calls, ["data:ok"])
        self.assertEqual(len(cache), 1)

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


def _verdict_record(verdict: str, **extra) -> dict:
    """判定回调的标准返回记录（verdict 三值 + 审计字段）。"""
    return {
        "verdict": verdict,
        "reason": f"{verdict} reason",
        "image_source": "poster",
        "asset_id": "stub",
        **extra,
    }


def _video_item(url: str, term: str, thumbnail: str | None = None) -> MaterialInfo:
    source = {"provider": "pexels", "search_term": term, "asset_id": url}
    if thumbnail is not None:
        source["thumbnail_url"] = thumbnail
    return MaterialInfo(provider="pexels", url=url, duration=10, source_info=source)


class TestMatchSegments(unittest.TestCase):
    """match_segments 三段漏斗主编排：配额、走查预算、降级链与去重契约。

    全 mock 表面：LLM（查询包）、搜索、重排、VLM 判定、下载、image-gen、
    缩略图（data URI）与嵌入向量（粗排）；断言调用次数与 SegmentMaterials
    字段（不以日志为成功依据），日志仅在任务指定的 grep 锚点处断言。
    """

    # 与 image_embedding.DEFAULT_DUPLICATE_THRESHOLD 无关：测试门用固定阈值。
    _GATE_THRESHOLD = 0.68

    @staticmethod
    def _queries_json(terms: list[str]) -> str:
        return json.dumps(
            {
                "terms": terms,
                "coarse_query": "A broad scene.",
                "fine_query": "A precise moment.",
            }
        )

    def _run(
        self,
        segments: list[dict],
        llm_payloads: list[str],
        pages_by_term: dict,
        judge=None,
        walk_limit: int = 10,
        rerank_enabled: bool = True,
        rerank=None,
        vectors: dict | None = None,
        subject: str = "panda",
        generate_image=None,
        gate: bool = False,
        embed_text_fails: bool = False,
        refine_result: str = "a refined cinematic scene prompt",
    ) -> SimpleNamespace:
        """以确定性桩运行 match_segments，返回全部调用记录与结果。"""
        searched: list[tuple[str, int]] = []
        saved_urls: list[str] = []
        judge_calls: list[tuple[str, str]] = []
        image_lock = threading.Lock()
        image_calls: list[tuple[dict, float, str, str]] = []
        refine_calls: list[tuple[str, str]] = []
        rerank_calls: list[tuple[str, list[str]]] = []
        captured: dict = {}
        llm_queue = list(llm_payloads)
        vectors = dict(vectors or {})

        def fake_llm(prompt):
            return llm_queue.pop(0)

        def fake_search(search_term, page=1, **_legacy_kwargs):
            # 两参新契约形态的桩：同时容忍旧四参调用（TypeError 回退链
            # 的第一跳在 match_segments 内，旧形态测试直接走第一跳）。
            searched.append((search_term, page))
            return list(pages_by_term.get((search_term, page), []))

        def fake_save_video(video_url, save_dir=""):
            saved_urls.append(video_url)
            return f"/saved/{video_url.rsplit('/', 1)[-1]}"

        def fake_judge(item, segment_text="", search_term=""):
            judge_calls.append((item.url, search_term))
            if callable(judge):
                return judge(item, segment_text, search_term)
            return _verdict_record(judge)

        def fake_rerank(query, items):
            rerank_calls.append((query, [i.url for i in items]))
            if rerank is None:
                return list(items)
            return rerank(query, items)

        def fake_refine(segment_text, subject_term=""):
            refine_calls.append((segment_text, subject_term))
            return refine_result

        def fake_image(segment, duration, refined_prompt, framing):
            # 并发安全（回填走线程池）：计数在锁内；clip 名由 framing 派生
            # 而非调用序（并发下调用序不确定，slot 序断言靠 framing 映射）。
            with image_lock:
                image_calls.append((segment, duration, refined_prompt, framing))
            return (
                f"/saved/gen-{framing.split()[0].lower()}.mp4",
                {"model": "Kwai-Kolors/Kolors", "source": "kolors", "framing": framing},
            )

        def fake_download_thumbnail(url):
            return (url.encode(), (640, 320))

        stub_embed = _StubEmbedImage(vectors)
        gate_obj = None
        gate_cache: dict[str, list[float]] = {}
        register_calls: list[str] = []
        if gate:
            gate_obj = EmbeddingGate(
                model="m",
                api_key="k",
                threshold=self._GATE_THRESHOLD,
                vector_cache=gate_cache,
            )
            original_register = gate_obj.register_accepted

            def spy_register(url):
                register_calls.append(url)
                return original_register(url)

            gate_obj.register_accepted = spy_register

        original_coarse = video_match.coarse_rank

        def spy_coarse(pool, coarse_query, vector_cache, embedding_gate):
            captured["pool"] = list(pool)
            return original_coarse(pool, coarse_query, vector_cache, embedding_gate)

        with (
            patch.object(
                video_match.llm, "generate_response", side_effect=fake_llm
            ),
            patch.object(
                video_match, "download_thumbnail_bytes", fake_download_thumbnail
            ),
            patch.object(video_match, "to_data_uri", lambda payload: payload.decode()),
            patch.object(
                image_embedding,
                "embed_text",
                return_value=None if embed_text_fails else [1.0, 0.0, 0.0],
            ),
            patch.object(image_embedding, "embed_image", stub_embed),
            patch.object(video_match, "coarse_rank", spy_coarse),
            patch.object(
                video_match.material_rerank,
                "is_rerank_enabled",
                return_value=rerank_enabled,
            ),
            patch.object(
                video_match.material_rerank, "_walk_limit", return_value=walk_limit
            ),
            patch.object(
                video_match.material_rerank, "rerank_candidates", fake_rerank
            ),
            patch.object(
                video_match.image_gen,
                "refine_scene_prompt",
                side_effect=fake_refine,
            ),
            patch.object(video_match, "logger") as mock_logger,
        ):
            results = video_match.match_segments(
                segments=segments,
                video_subject=subject,
                search_videos=fake_search,
                save_video=fake_save_video,
                video_aspect="9:16",
                clip_duration=3,
                judge_candidate=fake_judge if judge is not None else None,
                embedding_gate=gate_obj,
                generate_image=generate_image if generate_image is not None else fake_image,
            )

        return SimpleNamespace(
            results=results,
            searched=searched,
            saved=saved_urls,
            judge_calls=judge_calls,
            image_calls=image_calls,
            refine_calls=refine_calls,
            rerank_calls=rerank_calls,
            captured=captured,
            gate=gate_obj,
            gate_cache=gate_cache,
            register_calls=register_calls,
            logger=mock_logger,
        )

    def _info_messages(self, run: SimpleNamespace) -> list[str]:
        return [str(c.args[0]) for c in run.logger.info.call_args_list]

    def _warning_messages(self, run: SimpleNamespace) -> list[str]:
        return [str(c.args[0]) for c in run.logger.warning.call_args_list]

    def test_full_happy_path_fills_quota_with_early_exit(self):
        """D=12.816/W=3 → 配额 5：fine 序走查拿满即提前退出，判定次数
        ≤ walk 预算；采纳候选经 save_video + register_accepted 入册。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(6)
            ],
            ("panda one", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in (6, 7)
            ],
            ("panda two", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (8, 9)
            ],
            ("panda two", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (10, 11)
            ],
        }
        # 候选向量：e0 分量 a_i=0.95-0.05i 保证与查询 [1,0,0] 的余弦严格
        # 递减（粗排/精排序不受影响）；走查将采纳的 5 个候选各占一个独立
        # 方位角，两两余弦 ≤ ~0.58 < 0.68——互不为近重复。register_accepted
        # 修复后采纳向量真正入册，若候选两两近似（如全部躺在同一平面上）
        # 门会把后续候选判重拒收，配额断言就会被 image-gen 回填打破。
        walked = (11, 10, 9, 8, 7)
        azimuth = {i: 1.1 * n for n, i in enumerate(walked)}
        vectors = {}
        for i in range(12):
            a = 0.95 - 0.05 * i
            b = math.sqrt(1.0 - a * a)
            phi = azimuth.get(i, 0.0)
            vectors[f"img-{i}"] = [a, b * math.cos(phi), b * math.sin(phi)]
        # 钉死 2 页：夹具含第 2 页候选，live config 后续改 max_search_pages 也不翻转。
        random.seed(20260909)
        with patch.dict(config.material_rerank, {"max_search_pages": 2}):
            run = self._run(
                segments=[{"index": 0, "text": "panda", "duration": 12.816}],
                llm_payloads=[self._queries_json(["panda one", "panda two"])],
                pages_by_term=pages,
                judge="relevant",
                walk_limit=10,
                rerank=lambda _query, items: list(reversed(items)),
                vectors=vectors,
                gate=True,
            )

        result = run.results[0]
        # 配额 = max(3, len(plan(12.816, 3))) = 5，走查按 fine 逆序拿满。
        self.assertEqual(result.clips, [f"/saved/u{i}.mp4" for i in (11, 10, 9, 8, 7)])
        self.assertEqual(
            run.saved, [f"https://v.example/u{i}.mp4" for i in (11, 10, 9, 8, 7)]
        )
        # 提前退出：判定次数 = 配额 5 ≤ walk 预算 10；判定回带出处词条。
        self.assertEqual(
            run.judge_calls,
            [(f"https://v.example/u{i}.mp4", "panda two") for i in (11, 10, 9, 8)]
            + [("https://v.example/u7.mp4", "panda one")],
        )
        # 采纳即回调查重门注册（accepted-only 契约由 match 层保证）；
        # 粗排预热共享缓存：同 URL 全链路零重复嵌入。
        self.assertEqual(
            run.register_calls,
            [f"https://v.example/u{i}.mp4" for i in (11, 10, 9, 8, 7)],
        )
        self.assertEqual(len(run.gate_cache), 12)
        # 精排输入 = 粗排 top-30（12 个全量、余弦降序），恰好一次。
        self.assertEqual(
            run.rerank_calls,
            [("A precise moment.", [f"https://v.example/u{i}.mp4" for i in range(12)])],
        )
        self.assertEqual(run.image_calls, [])
        self.assertEqual(result.resolved_term, "panda two")
        self.assertEqual(result.fallback_level, "self")
        self.assertEqual(result.search_term, "panda one")
        self.assertEqual(
            [(a["term"], a["found"]) for a in result.search_attempts],
            [("panda one", True), ("panda two", True)],
        )
        self.assertEqual(len(result.vlm_filter), 5)

    def test_backfill_generates_one_clip_per_remaining_window(self):
        """(a)(b)(d)(k) 全走查拒绝：每个剩余计划窗口独立一次 generate_image，
        per-slot 时长 == 窗口时长，refined_prompt 全窗口共享一份（refine 恰好
        一次），景别按 slot 轮转，clips 按 slot 序追加，holes 为空，汇总行
        clips=5/5 照常落日志。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(2)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(2)}
        segment = {"index": 0, "text": "panda", "duration": 12.816}
        random.seed(20260909)
        run = self._run(
            segments=[segment],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge="irrelevant",
            vectors=vectors,
        )

        result = run.results[0]
        # D=12.816/W=3 → 窗口 [3,3,3,3,0.816]，配额 5；视频全军覆没 → 5 窗全回填。
        self.assertEqual(len(run.image_calls), 5)
        windows = [3.0, 3.0, 3.0, 3.0, 0.816]
        self.assertEqual(
            sorted(round(d, 6) for _, d, _, _ in run.image_calls),
            sorted(windows),
        )
        # refined_prompt 恰好一份共享（同串对象语义：全部相等且非空）。
        refined_prompts = {p for _, _, p, _ in run.image_calls}
        self.assertEqual(refined_prompts, {"a refined cinematic scene prompt"})
        # 景别按 slot 轮转（segment_index=0 不错位）。
        framings = [f for _, _, _, f in run.image_calls]
        self.assertEqual(
            sorted(framings), sorted(image_gen._BACKFILL_FRAMINGS[:5])
        )
        # clips 按 slot 序追加（桩的 clip 名由 framing 派生，slot ↔ framing
        # 一一对应：slot i → framings[i]）。
        self.assertEqual(
            result.clips,
            [
                f"/saved/gen-{image_gen._BACKFILL_FRAMINGS[i].split()[0].lower()}.mp4"
                for i in range(5)
            ],
        )
        self.assertEqual(result.holes, [])
        self.assertEqual(len(result.image_gen), 5)
        self.assertEqual(
            result.clip_sources[-1],
            {"url": "", "local_file": "gen-detail.mp4"},
        )
        # refine 每段恰好一次，用 video_match 本地的 subject（英文归一词）。
        self.assertEqual(run.refine_calls, [("panda", "panda")])
        # 审计行：回填总行 + 逐窗口 engaged 行 + 汇总行。
        info = self._info_messages(run)
        self.assertTrue(
            any(
                "video match: image-gen backfill windows=5 refine=once" in m
                for m in info
            )
        )
        engaged = [m for m in info if "image-gen fallback engaged" in m]
        self.assertEqual(len(engaged), 5)
        self.assertIn("slot=0", engaged[0])
        self.assertIn(f"framing={image_gen._BACKFILL_FRAMINGS[0]!r}", engaged[0])
        self.assertTrue(
            any("material resolution summary: clips=5/5" in m for m in info)
        )

    def test_backfill_mixed_case_extra_slots_use_last_window_duration(self):
        """(h) 混合情形（0 < 命中 < 窗口数 < 配额）：windows=[3,1]、1 个视频
        命中、配额 3 → 2 个回填 slot，时长 [1.0, 1.0]（尾窗 + 最后一窗兜底）。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(2)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(2)}

        def judge(item, segment_text, search_term):
            if item.url.endswith("u1.mp4"):
                return _verdict_record("irrelevant")
            return _verdict_record("relevant")

        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 4.0}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge=judge,
            vectors=vectors,
        )

        result = run.results[0]
        # D=4/W=3 → windows [3,1]，命中 1（u1 判 irrelevant），配额 3 →
        # remaining=2，tail=[1.0] → slots [1.0, 1.0]。
        self.assertEqual(sorted(d for _, d, _, _ in run.image_calls), [1.0, 1.0])
        self.assertEqual(
            result.clips,
            ["/saved/u0.mp4", "/saved/gen-wide.mp4", "/saved/gen-close-up.mp4"],
        )
        self.assertEqual(result.holes, [])
        self.assertTrue(
            any("image-gen backfill windows=2 refine=once" in m for m in self._info_messages(run))
        )

    def test_backfill_parallelism_capped_at_three(self):
        """(c) 5 个回填 slot 走 max_workers=3 线程池：线程安全计数器观测到的
        峰值并发 ≤ 3，且 ≥ 2（确实并行而非串行）。"""
        lock = threading.Lock()
        state = {"active": 0, "max_active": 0, "calls": 0}

        def slow_image(segment, duration, refined_prompt, framing):
            with lock:
                state["active"] += 1
                state["calls"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
            return (
                f"/saved/gen-{framing.split()[0].lower()}.mp4",
                {"source": "kolors", "framing": framing},
            )

        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 12.816}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term={},
            judge=None,
            generate_image=slow_image,
        )

        self.assertEqual(state["calls"], 5)
        self.assertLessEqual(state["max_active"], 3)
        self.assertGreaterEqual(state["max_active"], 2)
        self.assertEqual(len(run.results[0].clips), 5)

    def test_refine_once_per_segment_and_never_when_quota_filled(self):
        """(d) refine 每段至多一次；配额被视频素材拿满（remaining==0）时
        零 refine、零 generate_image。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(4)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(4)}
        random.seed(20260909)
        filled = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge="relevant",
            vectors=vectors,
        )
        # D=3.744 → 单窗，配额 3：3 个视频命中拿满 → 不进回填分支。
        self.assertEqual(filled.results[0].clips, [f"/saved/u{i}.mp4" for i in range(3)])
        self.assertEqual(filled.image_calls, [])
        self.assertEqual(filled.refine_calls, [])

        backfilled = self._run(
            segments=[
                {"index": 0, "text": "panda", "duration": 3.744},
                {"index": 1, "text": "second", "duration": 3.744},
            ],
            llm_payloads=[
                self._queries_json(["panda one"]),
                self._queries_json(["panda one"]),
            ],
            pages_by_term={},
            judge=None,
        )
        # 两段各自整段回填：每段恰好一次 refine（共 2 次），每次带本段旁白。
        self.assertEqual(len(backfilled.image_calls), 6)
        self.assertEqual(len(backfilled.refine_calls), 2)
        self.assertEqual(
            [text for text, _term in backfilled.refine_calls], ["panda", "second"]
        )

    def test_backfill_failure_tail_slot_records_hole(self):
        """(e) 尾部 slot 生成抛异常：该计划窗口记入 holes（= 已填窗口数 +
        slot），其余窗口 clip 完好；失败只隔离在本段，后续 segment 照常
        全量回填成功。"""
        pages = {}
        boom_framing = image_gen._BACKFILL_FRAMINGS[2]

        def exploding_image(segment, duration, refined_prompt, framing):
            if segment.get("index") == 0 and framing == boom_framing:
                raise RuntimeError("kolors exploded")
            return (
                f"/saved/gen-{framing.split()[0].lower()}.mp4",
                {"source": "kolors", "framing": framing},
            )

        run = self._run(
            segments=[
                {"index": 0, "text": "panda", "duration": 12.816},
                {"index": 1, "text": "second", "duration": 12.816},
            ],
            llm_payloads=[
                self._queries_json(["panda one"]),
                self._queries_json(["panda one"]),
            ],
            pages_by_term=pages,
            judge=None,
            generate_image=exploding_image,
        )

        result = run.results[0]
        # 5 窗全回填（已填 0），slot 2 失败 → hole 计划窗口 2；其余 4 个 clip。
        self.assertEqual(result.holes, [2])
        self.assertEqual(
            result.clips,
            [
                f"/saved/gen-{image_gen._BACKFILL_FRAMINGS[i].split()[0].lower()}.mp4"
                for i in (0, 1, 3, 4)
            ],
        )
        # 后续 segment 不受前段失败影响：5 窗全部成功，零 hole。
        self.assertEqual(run.results[1].holes, [])
        self.assertEqual(len(run.results[1].clips), 5)
        self.assertTrue(
            any(
                "image-gen backfill failed: slot=2 error=RuntimeError: kolors exploded"
                in m
                for m in self._warning_messages(run)
            )
        )
        self.assertTrue(
            any("material resolution summary: clips=4/5" in m for m in self._info_messages(run))
        )

    def test_refine_empty_marks_all_tail_windows_as_holes(self):
        """(f) refine 返回空串：全部尾部计划窗口记 holes，零 generate_image、
        零 clip 追加，流水线继续完成（fail-open）。"""
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 12.816}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term={},
            judge=None,
            refine_result="",
        )

        result = run.results[0]
        self.assertEqual(run.image_calls, [])
        self.assertEqual(run.refine_calls, [("panda", "panda")])
        self.assertEqual(result.holes, [0, 1, 2, 3, 4])
        self.assertEqual(result.clips, [])
        self.assertTrue(
            any("material resolution summary: clips=0/5" in m for m in self._info_messages(run))
        )

    def test_vlm_disabled_skips_search_and_backfills_whole_segment(self):
        """(g) VLM 关闭（judge=None）：零搜索、零下载，配额 = needed_clips
        个回填调用逐窗口覆盖全部计划窗口。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(4)
            ],
        }
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 12.816}],
            llm_payloads=[self._queries_json(["panda one", "panda two"])],
            pages_by_term=pages,
            judge=None,
        )

        result = run.results[0]
        self.assertEqual(run.searched, [])
        self.assertEqual(run.saved, [])
        self.assertEqual(run.judge_calls, [])
        # 配额 5 = needed_clips：5 次调用逐窗覆盖（时长多重集 == 窗口计划）。
        self.assertEqual(len(run.image_calls), 5)
        self.assertEqual(
            sorted(round(d, 6) for _, d, _, _ in run.image_calls),
            sorted([3.0, 3.0, 3.0, 3.0, 0.816]),
        )
        self.assertEqual(result.holes, [])
        self.assertEqual(len(result.clips), 5)
        self.assertEqual(result.fallback_level, "subject")
        self.assertEqual(result.resolved_term, "panda")
        self.assertEqual(result.search_attempts, [])
        self.assertTrue(
            any(
                "video match: vlm disabled, image-gen only" in m
                for m in self._info_messages(run)
            )
        )

    def test_backfill_failed_extra_slot_records_no_hole(self):
        """(i) 多样性名额 slot（slot ≥ len(tail)，超出计划窗口）失败：不记
        hole，其余 clip 完好，流水线照常完成。"""
        boom_framing = image_gen._BACKFILL_FRAMINGS[1]

        def exploding_image(segment, duration, refined_prompt, framing):
            if framing == boom_framing:
                raise RuntimeError("extra slot exploded")
            return (
                f"/saved/gen-{framing.split()[0].lower()}.mp4",
                {"source": "kolors", "framing": framing},
            )

        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(2)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(2)}

        def judge(item, segment_text, search_term):
            if item.url.endswith("u1.mp4"):
                return _verdict_record("irrelevant")
            return _verdict_record("relevant")

        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 4.0}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge=judge,
            vectors=vectors,
            generate_image=exploding_image,
        )

        result = run.results[0]
        # windows=[3,1]、命中 1 → tail=[1.0] 长 1，slot 1 是名额多余部分；
        # slot 1 失败 → 不对应任何计划窗口 → holes 为空。
        self.assertEqual(result.holes, [])
        self.assertEqual(result.clips, ["/saved/u0.mp4", "/saved/gen-wide.mp4"])
        self.assertTrue(
            any(
                "image-gen backfill failed: slot=1 error=RuntimeError: extra slot exploded"
                in m
                for m in self._warning_messages(run)
            )
        )
        self.assertTrue(
            any("material resolution summary: clips=2/3" in m for m in self._info_messages(run))
        )

    def test_backfill_empty_tail_uses_last_window_duration(self):
        """(j) 空尾（视频拿满全部窗口、名额仍差）：剩余调用全部用最后一窗
        时长，holes 为空。"""
        # D=3.744 → 单窗 [3.744]，配额 3：仅 1 个候选 → 命中 1、remaining=2、
        # tail=[] → 2 个 slot 全部 3.744。
        pages = {
            ("panda one", 1): [
                _video_item("https://v.example/u0.mp4", "panda one", "img-0"),
            ],
        }
        vectors = {"img-0": _vec_with_cos(0.9)}
        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge="relevant",
            vectors=vectors,
        )

        result = run.results[0]
        self.assertEqual(len(run.image_calls), 2)
        self.assertEqual(
            [d for _, d, _, _ in run.image_calls], [3.744, 3.744]
        )
        self.assertEqual(
            result.clips,
            ["/saved/u0.mp4", "/saved/gen-wide.mp4", "/saved/gen-close-up.mp4"],
        )
        self.assertEqual(result.holes, [])

    def test_coarse_failure_flows_interleave_pool_to_fine(self):
        """粗排查询向量不可得：fail-open 返回 interleave pool[:30]，精排
        与走查照常完成——流水线绝不因粗排失败阻塞。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in (0, 1, 2)
            ],
            ("panda two", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (3, 4, 5)
            ],
            ("panda one", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in (6, 7)
            ],
            ("panda two", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (8, 9)
            ],
        }
        # 钉死 2 页：interleave 池含第 2 页候选，live config 改值不翻转。
        random.seed(20260909)
        with patch.dict(config.material_rerank, {"max_search_pages": 2}):
            run = self._run(
                segments=[{"index": 0, "text": "panda", "duration": 12.816}],
                llm_payloads=[self._queries_json(["panda one", "panda two"])],
                pages_by_term=pages,
                judge="relevant",
                walk_limit=10,
                embed_text_fails=True,
            )

        result = run.results[0]
        # T2 池 shuffle 后的种子序（random.seed(20260909)）：u0..u9 的固定
        # 置换；粗排 fail-open 时精排收到的仍是完整池（不缩水）。
        shuffled = [
            f"https://v.example/u{i}.mp4"
            for i in (7, 8, 5, 3, 0, 4, 6, 2, 9, 1)
        ]
        self.assertEqual([c["url"] for c in run.captured["pool"]], shuffled)
        self.assertEqual(run.rerank_calls[0][1], shuffled)
        self.assertEqual(
            result.clips, [f"/saved/u{i}.mp4" for i in (7, 8, 5, 3, 0)]
        )
        self.assertEqual(len(run.judge_calls), 5)
        self.assertTrue(
            any(
                "video match: coarse rank failed, fail-open" in m
                for m in self._warning_messages(run)
            )
        )

    def test_fine_disabled_walks_coarse_order_capped(self):
        """[material_rerank] enabled=false：精排整段跳过（零调用），走查
        消费粗排序并被 walk 预算截断。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(6)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(6)}
        # 固定随机序：T2 引入池 shuffle 后，粗排序断言仍确定。
        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge="relevant",
            walk_limit=3,
            rerank_enabled=False,
            vectors=vectors,
        )

        result = run.results[0]
        self.assertEqual(run.rerank_calls, [])
        self.assertEqual(result.clips, ["/saved/u0.mp4", "/saved/u1.mp4", "/saved/u2.mp4"])
        self.assertEqual(len(run.judge_calls), 3)
        self.assertTrue(
            any(
                "video match: fine rerank disabled, using coarse order" in m
                for m in self._info_messages(run)
            )
        )

    def test_fine_fail_open_uses_coarse_order(self):
        """精排调用本身抛异常：告警后按粗排序走查，cap 仍然生效。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(6)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(6)}

        def exploding_rerank(query, items):
            raise RuntimeError("reranker exploded")

        # 固定随机序：T2 引入池 shuffle 后，粗排序断言仍确定。
        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge="relevant",
            walk_limit=3,
            rerank=exploding_rerank,
            vectors=vectors,
        )

        result = run.results[0]
        self.assertEqual(result.clips, ["/saved/u0.mp4", "/saved/u1.mp4", "/saved/u2.mp4"])
        self.assertEqual(len(run.judge_calls), 3)
        self.assertTrue(
            any(
                "video match: fine rerank failed" in m
                for m in self._warning_messages(run)
            )
        )

    def test_duplicate_url_across_terms_appears_once_in_pool(self):
        """同一 URL 被两个词条同时返回：按 URL 去重保首个（interleave 序
        中先出现的词条持有该候选）。"""
        pages = {
            ("panda one", 1): [
                _video_item("https://v.example/dup.mp4", "panda one", "img-dup"),
                _video_item("https://v.example/a.mp4", "panda one", "img-a"),
            ],
            ("panda two", 1): [
                _video_item("https://v.example/dup.mp4", "panda two", "img-dup"),
                _video_item("https://v.example/b.mp4", "panda two", "img-b"),
            ],
        }
        # 固定随机序：T2 引入池 shuffle 后，pool 顺序断言仍确定。
        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one", "panda two"])],
            pages_by_term=pages,
            judge="relevant",
        )

        pool = run.captured["pool"]
        # T2 池 shuffle 后的种子序（random.seed(20260909)）：[dup,a,b] 的
        # 固定置换；dup 的持有者仍是 panda one（去重保首个不受打乱影响）。
        self.assertEqual(
            [c["url"] for c in pool],
            [
                "https://v.example/a.mp4",
                "https://v.example/b.mp4",
                "https://v.example/dup.mp4",
            ],
        )
        self.assertEqual(pool[0]["term"], "panda one")
        # 去重后配额照常拿满，dup URL 只下载一次。
        self.assertEqual(
            run.results[0].clips,
            ["/saved/a.mp4", "/saved/b.mp4", "/saved/dup.mp4"],
        )
        self.assertEqual(run.saved.count("https://v.example/dup.mp4"), 1)
        self.assertEqual(len(run.judge_calls), 3)

    def test_used_url_never_enters_later_pool(self):
        """前面 segment 已采纳的 URL：后续 segment 的候选池不含它，不重复
        判定、不重复下载（跨段 used 排除镜像旧机制）。"""
        pages = {
            ("t one", 1): [
                _video_item("https://v.example/x.mp4", "t one", "img-x"),
                _video_item("https://v.example/a.mp4", "t one", "img-a"),
                _video_item("https://v.example/b.mp4", "t one", "img-b"),
            ],
            ("t two", 1): [
                _video_item("https://v.example/x.mp4", "t two", "img-x"),
                _video_item("https://v.example/c.mp4", "t two", "img-c"),
                _video_item("https://v.example/d.mp4", "t two", "img-d"),
                _video_item("https://v.example/e.mp4", "t two", "img-e"),
            ],
        }
        # 固定随机序：T2 引入池 shuffle 后，逐段 clips 顺序断言仍确定。
        random.seed(20260909)
        run = self._run(
            segments=[
                {"index": 0, "text": "first", "duration": 3.744},
                {"index": 1, "text": "second", "duration": 3.744},
            ],
            llm_payloads=[
                self._queries_json(["t one"]),
                self._queries_json(["t two"]),
            ],
            pages_by_term=pages,
            judge="relevant",
        )

        # T2 池 shuffle 后的种子序（random.seed(20260909)）：两段 clips 各是
        # 其打乱后池序的头部切片（[x,a,b]→[a,b,x]，[c,d,e]→[d,e,c]）。
        self.assertEqual(
            run.results[0].clips,
            ["/saved/a.mp4", "/saved/b.mp4", "/saved/x.mp4"],
        )
        self.assertEqual(
            run.results[1].clips,
            ["/saved/d.mp4", "/saved/e.mp4", "/saved/c.mp4"],
        )
        # x 全程只下载一次、只判定一次（seg1 的池里根本没有它）。
        self.assertEqual(run.saved.count("https://v.example/x.mp4"), 1)
        self.assertEqual(
            [url for url, _term in run.judge_calls].count("https://v.example/x.mp4"),
            1,
        )
        self.assertNotIn(
            "https://v.example/x.mp4", [c["url"] for c in run.captured["pool"]]
        )

    def test_search_cache_same_term_page_once_across_segments(self):
        """两段生成同一词条：(词条, 页) 备忘使命中缓存的关键词组合只透传
        一次供应商调用；第二段从剩余新鲜候选拿满名额。"""
        pages = {
            ("city walk", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "city walk", f"img-{i}")
                for i in range(6)
            ],
        }
        # 钉死 2 页：断言精确到 (词条, 页) 序列，live config 改值不翻转。
        random.seed(20260909)
        with patch.dict(config.material_rerank, {"max_search_pages": 2}):
            run = self._run(
                segments=[
                    {"index": 0, "text": "walk one", "duration": 3.744},
                    {"index": 1, "text": "walk two", "duration": 3.744},
                ],
                llm_payloads=[
                    self._queries_json(["city walk"]),
                    self._queries_json(["city walk"]),
                ],
                pages_by_term=pages,
                judge="relevant",
            )

        self.assertEqual(
            run.searched, [("city walk", 1), ("city walk", 2)]
        )
        # 搜索边界契约：search_videos 收到的词条 == LLM 干净词条（无主题
        # 后缀，无双重追加）。
        self.assertEqual(
            sorted({term for term, _page in run.searched}), ["city walk"]
        )
        # T2 池 shuffle 后的种子序（random.seed(20260909)）：两段 clips 各是
        # 其打乱后池序的头部切片（seg0 池 u0..u5 → [u3,u4,u2]；seg1 剩余
        # {u0,u1,u5} → [u5,u0,u1]）。
        self.assertEqual(
            run.results[0].clips,
            ["/saved/u3.mp4", "/saved/u4.mp4", "/saved/u2.mp4"],
        )
        self.assertEqual(
            run.results[1].clips,
            ["/saved/u5.mp4", "/saved/u0.mp4", "/saved/u1.mp4"],
        )
        self.assertEqual(len(run.saved), 6)

    def test_max_search_pages_one_limits_search_to_first_page(self):
        """[material_rerank] max_search_pages=1：每个词条只抓第 1 页，第 2 页
        永不透传（页数可配置契约的集成面；复用跨段备忘的夹具形状）。"""
        pages = {
            ("city walk", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "city walk", f"img-{i}")
                for i in range(6)
            ],
        }
        random.seed(20260909)
        with patch.dict(config.material_rerank, {"max_search_pages": 1}):
            run = self._run(
                segments=[
                    {"index": 0, "text": "walk one", "duration": 3.744},
                    {"index": 1, "text": "walk two", "duration": 3.744},
                ],
                llm_payloads=[
                    self._queries_json(["city walk"]),
                    self._queries_json(["city walk"]),
                ],
                pages_by_term=pages,
                judge="relevant",
            )

        self.assertEqual(run.searched, [("city walk", 1)])
        # T2 池 shuffle 后的种子序（random.seed(20260909)）：与跨段备忘测试
        # 同一夹具形状，同一打乱结果（seg0 [u3,u4,u2]；seg1 [u5,u0,u1]）。
        self.assertEqual(
            run.results[0].clips,
            ["/saved/u3.mp4", "/saved/u4.mp4", "/saved/u2.mp4"],
        )
        self.assertEqual(
            run.results[1].clips,
            ["/saved/u5.mp4", "/saved/u0.mp4", "/saved/u1.mp4"],
        )
        self.assertEqual(len(run.saved), 6)

    def test_quota_accounting_pinned_to_window_plan(self):
        """配额 = max(CLIPS_PER_SEGMENT, len(segment_window_plan(D, W)))：
        D=12.816/W=3 → 5 窗；D=9.48/W=3 → 3 窗。名额满即停走（判定次数 =
        配额，而非 walk 预算或池大小）。"""
        for duration, expected in ((12.816, 5), (9.48, 3)):
            with self.subTest(duration=duration):
                pages = {
                    ("panda one", 1): [
                        _video_item(
                            f"https://v.example/u{i}.mp4", "panda one", f"img-{i}"
                        )
                        for i in range(8)
                    ],
                }
                vectors = {
                    f"img-{i}": _vec_with_cos(0.9 - 0.05 * i) for i in range(8)
                }
                run = self._run(
                    segments=[{"index": 0, "text": "panda", "duration": duration}],
                    llm_payloads=[self._queries_json(["panda one"])],
                    pages_by_term=pages,
                    judge="relevant",
                    walk_limit=10,
                    vectors=vectors,
                )
                self.assertEqual(len(run.results[0].clips), expected)
                self.assertEqual(len(run.judge_calls), expected)
                self.assertEqual(len(run.saved), expected)

    def test_judge_exception_skips_candidate_and_continues(self):
        """单候选判定抛异常 = 跳过该候选继续走查（fail-open），绝不阻塞。"""
        pages = {
            ("panda one", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(4)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(4)}

        def judge(item, segment_text, search_term):
            if item.url.endswith("u1.mp4"):
                raise RuntimeError("vlm exploded")
            return _verdict_record("relevant")

        # 固定随机序：T2 引入池 shuffle 后，clips 顺序断言仍确定。
        random.seed(20260909)
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge=judge,
            vectors=vectors,
        )

        result = run.results[0]
        # u1 异常被跳过，u2/u3 依次补位拿满配额 3。
        self.assertEqual(
            result.clips, ["/saved/u0.mp4", "/saved/u2.mp4", "/saved/u3.mp4"]
        )
        self.assertEqual(len(run.judge_calls), 4)
        self.assertTrue(
            any("vlm judge failed, fail-open" in m for m in self._warning_messages(run))
        )

    def test_empty_pool_and_failed_queries_backfill_whole_segment(self):
        """畸形输入：LLM 查询包全失败 + 搜索空手 → 主题词兜底搜索、空池
        直接整段回填（每窗一调用），流水线不崩、不阻塞。"""
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=["total garbage", "still garbage"],
            pages_by_term={},
            judge="relevant",
        )

        result = run.results[0]
        # 词条为空 → 主题词兜底；第 1 页即空 → 不翻第 2 页。
        self.assertEqual(run.searched, [("panda", 1)])
        self.assertEqual(run.judge_calls, [])
        # D=3.744 → 单窗 [3.744]，配额 3，已填 0 → 3 个 slot 全部 3.744。
        self.assertEqual(len(run.image_calls), 3)
        self.assertEqual(
            [d for _, d, _, _ in run.image_calls], [3.744, 3.744, 3.744]
        )
        self.assertEqual(
            result.clips,
            [
                "/saved/gen-wide.mp4",
                "/saved/gen-close-up.mp4",
                "/saved/gen-low-angle.mp4",
            ],
        )
        self.assertEqual(result.holes, [])
        self.assertEqual(result.fallback_level, "subject")
        self.assertEqual(
            result.search_attempts,
            [{"level": "self", "term": "panda", "found": False}],
        )

    def test_zero_duration_segment_skips_search_and_backfill(self):
        """(l) D=0（异常旁白）：窗口计划为空；match_segments 跳过查询/搜索/
        回填，零 generate_image、零 holes、空 SegmentMaterials——不得经
        last_window 兜底为不存在的计划窗口生成素材。装配层对同时缺素材与
        时长的段同样跳过时间线（video.py:919-924），两端语义对称。"""
        self.assertEqual(segment_window_plan(0, 3), [])
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 0}],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term={},
            judge=None,
        )

        self.assertEqual(len(run.results), 1)
        result = run.results[0]
        self.assertEqual(run.image_calls, [])
        self.assertEqual(run.searched, [])
        self.assertEqual(run.saved, [])
        self.assertEqual(result.clips, [])
        self.assertEqual(result.clip_sources, [])
        self.assertEqual(result.image_gen, [])
        self.assertEqual(result.holes, [])
        self.assertEqual(result.search_term, "")
        self.assertEqual(result.resolved_term, "")
        self.assertEqual(result.fallback_level, "")
        self.assertEqual(result.search_attempts, [])
        self.assertTrue(
            any(
                "video match: non-positive duration" in m
                for m in self._warning_messages(run)
            )
        )


if __name__ == "__main__":
    unittest.main()
