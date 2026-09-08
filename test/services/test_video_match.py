import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import MaterialInfo
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
    ) -> SimpleNamespace:
        """以确定性桩运行 match_segments，返回全部调用记录与结果。"""
        searched: list[tuple[str, int]] = []
        saved_urls: list[str] = []
        judge_calls: list[tuple[str, str]] = []
        image_calls: list[tuple[dict, float]] = []
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

        def fake_image(segment, duration):
            image_calls.append((segment, duration))
            return (
                f"/saved/gen-{segment.get('index', 0)}.mp4",
                {"model": "Kwai-Kolors/Kolors", "source": "kolors"},
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
            ("panda one panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(6)
            ],
            ("panda one panda", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in (6, 7)
            ],
            ("panda two panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (8, 9)
            ],
            ("panda two panda", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (10, 11)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.95 - 0.05 * i) for i in range(12)}
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
            [(f"https://v.example/u{i}.mp4", "panda two panda") for i in (11, 10, 9, 8)]
            + [("https://v.example/u7.mp4", "panda one panda")],
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
        self.assertEqual(result.resolved_term, "panda two panda")
        self.assertEqual(result.fallback_level, "self")
        self.assertEqual(result.search_term, "panda one panda")
        self.assertEqual(
            [(a["term"], a["found"]) for a in result.search_attempts],
            [("panda one panda", True), ("panda two panda", True)],
        )
        self.assertEqual(len(result.vlm_filter), 5)

    def test_partial_fill_backfills_remaining_windows(self):
        """部分命中：image-gen 回填一次覆盖未填充尾部窗口，时长 = 尾窗和；
        SegmentMaterials 同时反映视频 clip 与生成 clip 的来源。"""
        pages = {
            ("panda one panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(4)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(4)}

        def judge(item, segment_text, search_term):
            if item.url.endswith(("u2.mp4", "u3.mp4")):
                return _verdict_record("irrelevant")
            return _verdict_record("relevant")

        segment = {"index": 0, "text": "panda", "duration": 12.816}
        run = self._run(
            segments=[segment],
            llm_payloads=[self._queries_json(["panda one"])],
            pages_by_term=pages,
            judge=judge,
            vectors=vectors,
        )

        result = run.results[0]
        # 配额 5，视频命中 2（u2/u3 判 irrelevant），尾部窗口 [3,3,0.816]。
        self.assertEqual(len(run.judge_calls), 4)
        self.assertEqual(len(run.image_calls), 1)
        segment_arg, duration_arg = run.image_calls[0]
        self.assertEqual(segment_arg, segment)
        self.assertAlmostEqual(duration_arg, 6.816, places=6)
        self.assertEqual(
            result.clips, ["/saved/u0.mp4", "/saved/u1.mp4", "/saved/gen-0.mp4"]
        )
        self.assertEqual(
            result.clip_sources[-1], {"url": "", "local_file": "gen-0.mp4"}
        )
        self.assertEqual(result.image_gen, [{"model": "Kwai-Kolors/Kolors", "source": "kolors"}])
        self.assertEqual(result.resolved_term, "panda one panda")
        self.assertEqual(result.fallback_level, "self")
        self.assertEqual(len(result.vlm_filter), 4)

    def test_vlm_disabled_skips_search_and_backfills_whole_segment(self):
        """VLM 关闭（judge=None）：零搜索、零下载，整段 image-gen（强制立场）。"""
        pages = {
            ("panda one panda", 1): [
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
        # 整段回填：时长 = 全部窗口之和 ≈ 段时长。
        self.assertEqual(len(run.image_calls), 1)
        segment_arg, duration_arg = run.image_calls[0]
        self.assertEqual(segment_arg["index"], 0)
        self.assertAlmostEqual(duration_arg, 12.816, places=3)
        self.assertEqual(result.clips, ["/saved/gen-0.mp4"])
        self.assertEqual(result.fallback_level, "subject")
        self.assertEqual(result.resolved_term, "panda")
        self.assertEqual(result.search_attempts, [])
        self.assertTrue(
            any(
                "video match: vlm disabled, image-gen only" in m
                for m in self._info_messages(run)
            )
        )

    def test_coarse_failure_flows_interleave_pool_to_fine(self):
        """粗排查询向量不可得：fail-open 返回 interleave pool[:30]，精排
        与走查照常完成——流水线绝不因粗排失败阻塞。"""
        pages = {
            ("panda one panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in (0, 1, 2)
            ],
            ("panda two panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (3, 4, 5)
            ],
            ("panda one panda", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in (6, 7)
            ],
            ("panda two panda", 2): [
                _video_item(f"https://v.example/u{i}.mp4", "panda two", f"img-{i}")
                for i in (8, 9)
            ],
        }
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 12.816}],
            llm_payloads=[self._queries_json(["panda one", "panda two"])],
            pages_by_term=pages,
            judge="relevant",
            walk_limit=10,
            embed_text_fails=True,
        )

        result = run.results[0]
        interleave = [f"https://v.example/u{i}.mp4" for i in range(10)]
        self.assertEqual([c["url"] for c in run.captured["pool"]], interleave)
        # 精排收到的仍是完整 interleave 池（粗排 fail-open 不缩水）。
        self.assertEqual(run.rerank_calls[0][1], interleave)
        self.assertEqual(result.clips, [f"/saved/u{i}.mp4" for i in range(5)])
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
            ("panda one panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(6)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(6)}
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
            ("panda one panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(6)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(6)}

        def exploding_rerank(query, items):
            raise RuntimeError("reranker exploded")

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
            ("panda one panda", 1): [
                _video_item("https://v.example/dup.mp4", "panda one", "img-dup"),
                _video_item("https://v.example/a.mp4", "panda one", "img-a"),
            ],
            ("panda two panda", 1): [
                _video_item("https://v.example/dup.mp4", "panda two", "img-dup"),
                _video_item("https://v.example/b.mp4", "panda two", "img-b"),
            ],
        }
        run = self._run(
            segments=[{"index": 0, "text": "panda", "duration": 3.744}],
            llm_payloads=[self._queries_json(["panda one", "panda two"])],
            pages_by_term=pages,
            judge="relevant",
        )

        pool = run.captured["pool"]
        self.assertEqual(
            [c["url"] for c in pool],
            [
                "https://v.example/dup.mp4",
                "https://v.example/a.mp4",
                "https://v.example/b.mp4",
            ],
        )
        self.assertEqual(pool[0]["term"], "panda one panda")
        # 去重后配额照常拿满，dup URL 只下载一次。
        self.assertEqual(
            run.results[0].clips,
            ["/saved/dup.mp4", "/saved/a.mp4", "/saved/b.mp4"],
        )
        self.assertEqual(run.saved.count("https://v.example/dup.mp4"), 1)
        self.assertEqual(len(run.judge_calls), 3)

    def test_used_url_never_enters_later_pool(self):
        """前面 segment 已采纳的 URL：后续 segment 的候选池不含它，不重复
        判定、不重复下载（跨段 used 排除镜像旧机制）。"""
        pages = {
            ("t one panda", 1): [
                _video_item("https://v.example/x.mp4", "t one", "img-x"),
                _video_item("https://v.example/a.mp4", "t one", "img-a"),
                _video_item("https://v.example/b.mp4", "t one", "img-b"),
            ],
            ("t two panda", 1): [
                _video_item("https://v.example/x.mp4", "t two", "img-x"),
                _video_item("https://v.example/c.mp4", "t two", "img-c"),
                _video_item("https://v.example/d.mp4", "t two", "img-d"),
                _video_item("https://v.example/e.mp4", "t two", "img-e"),
            ],
        }
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

        self.assertEqual(
            run.results[0].clips,
            ["/saved/x.mp4", "/saved/a.mp4", "/saved/b.mp4"],
        )
        self.assertEqual(
            run.results[1].clips,
            ["/saved/c.mp4", "/saved/d.mp4", "/saved/e.mp4"],
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
            ("city walk panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "city walk", f"img-{i}")
                for i in range(6)
            ],
        }
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
            run.searched, [("city walk panda", 1), ("city walk panda", 2)]
        )
        self.assertEqual(
            run.results[0].clips,
            ["/saved/u0.mp4", "/saved/u1.mp4", "/saved/u2.mp4"],
        )
        self.assertEqual(
            run.results[1].clips,
            ["/saved/u3.mp4", "/saved/u4.mp4", "/saved/u5.mp4"],
        )
        self.assertEqual(len(run.saved), 6)

    def test_quota_accounting_pinned_to_window_plan(self):
        """配额 = max(CLIPS_PER_SEGMENT, len(segment_window_plan(D, W)))：
        D=12.816/W=3 → 5 窗；D=9.48/W=3 → 3 窗。名额满即停走（判定次数 =
        配额，而非 walk 预算或池大小）。"""
        for duration, expected in ((12.816, 5), (9.48, 3)):
            with self.subTest(duration=duration):
                pages = {
                    ("panda one panda", 1): [
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
            ("panda one panda", 1): [
                _video_item(f"https://v.example/u{i}.mp4", "panda one", f"img-{i}")
                for i in range(4)
            ],
        }
        vectors = {f"img-{i}": _vec_with_cos(0.9 - 0.1 * i) for i in range(4)}

        def judge(item, segment_text, search_term):
            if item.url.endswith("u1.mp4"):
                raise RuntimeError("vlm exploded")
            return _verdict_record("relevant")

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
        直接整段回填，流水线不崩、不阻塞。"""
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
        self.assertEqual(len(run.image_calls), 1)
        self.assertAlmostEqual(run.image_calls[0][1], 3.744, places=3)
        self.assertEqual(result.clips, ["/saved/gen-0.mp4"])
        self.assertEqual(result.fallback_level, "subject")
        self.assertEqual(
            result.search_attempts,
            [{"level": "self", "term": "panda", "found": False}],
        )


if __name__ == "__main__":
    unittest.main()
