"""Library-first selection mode tests (`[app] coarse_rank_mode = "library"`).

Covers the pieces that changed for the library-first matching plan:

- `_select_by_library_order`: pool order is the selection order, the dedup
  gate runs lazily per judged candidate (no pool-wide embedding), and the
  gate-absent path truncates like `coarse_rank` does;
- `match_segments` library mode: the pool keeps library order (no shuffle),
  the segment's coarse_query leads the search terms, and candidates carry no
  preview image until the gate asks for one;
- the dedup gate service backend: vectors come from the retrieval service
  `POST /embed`, failures fail open, and DashScope is never touched.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo
from app.services import image_embedding, video_match
from app.services.image_embedding import EmbeddingGate
from app.services.video_match import SegmentQueries


def _item(pexels_id: int, url: str | None = None) -> MaterialInfo:
    """A library-shaped candidate: source_info carries asset_id + thumbnail."""
    return MaterialInfo(
        url=url or f"https://videos.pexels.com/video-files/{pexels_id}/a-{pexels_id}.mp4",
        provider="library",
        duration=10,
        width=1080,
        height=1920,
        source_info={
            "asset_id": str(pexels_id),
            "search_term": "theme",
            "thumbnail_url": f"https://images.pexels.com/t/{pexels_id}.jpeg",
            "source_page": f"https://www.pexels.com/video/x-{pexels_id}/",
        },
    )


class TestSelectByLibraryOrder(unittest.TestCase):
    """`_select_by_library_order`: 池序即选材序，门走查按需嵌入。"""

    def setUp(self):
        self.gate = EmbeddingGate(
            model="m", api_key="k", threshold=0.9, vector_cache={}
        )
        self.embeds: list[str] = []
        patcher = patch.object(
            image_embedding,
            "embed_via_service",
            side_effect=lambda asset_id: self.embeds.append(asset_id) or [1.0, 0.0],
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # backend = service：门经检索服务取向量；预览图同样本地化，本测试不触网。
        self._gate_cfg = dict(config.image_embedding)
        config.image_embedding["backend"] = "service"
        fetch_patcher = patch.object(
            video_match, "fetch_image_bytes", return_value=(b"jpeg", (640, 320))
        )
        fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)

    def tearDown(self):
        config.image_embedding.clear()
        config.image_embedding.update(self._gate_cfg)

    def test_pool_order_is_preserved(self):
        pool = [
            {"url": f"u{i}", "asset_id": f"{i}", "item": _item(i), "term": "t"}
            for i in range(3)
        ]
        selected, skips = video_match._select_by_library_order(pool, self.gate)
        self.assertEqual([c["url"] for c in selected], ["u0", "u1", "u2"])
        self.assertEqual(skips, [])
        # 门按需嵌入：每个被选中的候选恰好一次服务调用。
        self.assertEqual(self.embeds, ["0", "1", "2"])

    def test_duplicates_against_earlier_segments_are_rejected(self):
        # 门只与「已采纳」注册表比对（跨段语义）：先前段采纳过的素材，本段
        # 的近重复候选在选材边界就被拒收。
        self.gate._accepted["earlier"] = [1.0, 0.0]
        vectors = {"0": [0.99, 0.14], "1": [0.97, 0.24], "2": [0.0, 1.0]}
        patcher = patch.object(
            image_embedding,
            "embed_via_service",
            side_effect=lambda asset_id: self.embeds.append(asset_id) or vectors[asset_id],
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        pool = [
            {"url": f"u{i}", "asset_id": f"{i}", "item": _item(i), "term": "t"}
            for i in range(3)
        ]
        selected, skips = video_match._select_by_library_order(pool, self.gate)
        self.assertEqual([c["url"] for c in selected], ["u2"])
        self.assertEqual(len(skips), 2)
        self.assertTrue(all(s["reason"].startswith("cos=") for s in skips))

    def test_gate_service_failure_fails_open(self):
        # 服务后端失败：门放行（不回落 dashscope，A/B 两路径保持可区分），
        # VLM 走查是终审。embed_image 装炸弹证明绝无静默直连。
        patcher = patch.object(
            image_embedding,
            "embed_via_service",
            return_value=None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        def _bomb(data_uri, model, api_key, base_url):
            raise AssertionError("service backend must not fall back to DashScope")

        bomb = patch.object(image_embedding, "embed_image", side_effect=_bomb)
        bomb.start()
        self.addCleanup(bomb.stop)
        pool = [{"url": "u0", "asset_id": "0", "item": _item(0), "term": "t"}]
        selected, _ = video_match._select_by_library_order(pool, self.gate)
        self.assertEqual([c["url"] for c in selected], ["u0"])

    def test_service_backend_never_fetches_a_preview(self):
        # service 后端向量按 asset_id 取，预览图不再是判定输入：被拒收的
        # 候选也不为它白白取一张缩略图。
        fetch = patch.object(
            video_match, "fetch_image_bytes", side_effect=AssertionError("no preview needed")
        )
        fetch.start()
        self.addCleanup(fetch.stop)
        pool = [
            {"url": f"u{i}", "asset_id": f"{i}", "item": _item(i), "term": "t"}
            for i in range(3)
        ]
        selected, _ = video_match._select_by_library_order(pool, self.gate)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(c.get("data_uri", "") == "" for c in selected))

    def test_without_gate_truncates_pool(self):
        pool = [{"url": f"u{i}", "asset_id": f"{i}", "item": _item(i), "term": "t"} for i in range(40)]
        selected, skips = video_match._select_by_library_order(pool, None)
        self.assertEqual(len(selected), video_match._COARSE_TOP_K)
        self.assertEqual(skips, [])


class TestLibraryFirstMatchingMode(unittest.TestCase):
    """`[app] coarse_rank_mode = "library"`：池序保留、整句先进池、预览图懒取。"""

    def setUp(self):
        self._app_config = dict(config.app)
        config.app["coarse_rank_mode"] = "library"

    def tearDown(self):
        config.app.clear()
        config.app.update(self._app_config)

    def test_mode_helpers_read_config(self):
        self.assertTrue(video_match._library_first())
        self.assertEqual(video_match._coarse_rank_mode(), "library")

        config.app["coarse_rank_mode"] = "embed"
        self.assertFalse(video_match._library_first())

        config.app["coarse_rank_mode"] = "bogus"
        self.assertFalse(video_match._library_first())  # 未知值回落 embed

    def test_candidate_carries_no_preview_in_library_mode(self):
        item = _item(7)
        candidate = video_match._candidate_from_item(item, "t")
        self.assertEqual(candidate["data_uri"], "")

        with patch.object(video_match, "fetch_image_bytes") as fetch:
            fetch.return_value = (b"jpeg", (640, 320))
            data_uri = video_match._ensure_data_uri(candidate)

        self.assertNotEqual(data_uri, "")
        self.assertEqual(candidate["data_uri"], data_uri)
        fetch.assert_called_once()  # 只取一次，回写后第二次调用零取
        again = video_match._ensure_data_uri(candidate)
        self.assertEqual(again, data_uri)
        fetch.assert_called_once()

    def test_embed_mode_still_prebuilds_preview(self):
        config.app["coarse_rank_mode"] = "embed"
        with patch.object(video_match, "fetch_image_bytes") as fetch:
            fetch.return_value = (b"jpeg", (640, 320))
            candidate = video_match._candidate_from_item(_item(7), "t")
        fetch.assert_called_once()
        self.assertNotEqual(candidate["data_uri"], "")


class TestLibraryModeEndToEnd(unittest.TestCase):
    """match_segments 库优先模式的编排级行为：整句先搜、库序保留、零嵌入。

    ADR-0019 决策 1 的三件套都在这一层生效（此前只有 _select_by_library_order
    被测过）：coarse_query 随角度词条进池且排最前、random.shuffle 被跳过、
    粗排嵌入完全退场（embed_text / embed_image / fetch_image_bytes 全程装炸弹）。
    """

    def setUp(self):
        self._app_config = dict(config.app)
        config.app["coarse_rank_mode"] = "library"

    def tearDown(self):
        config.app.clear()
        config.app.update(self._app_config)

    def test_library_flow_leads_with_the_sentence_and_keeps_pool_order(self):
        queries = SegmentQueries(
            terms=["angle one", "angle two"],
            coarse_query="A broad scene.",
            fine_query="A precise moment.",
        )
        pool = [_item(i, url=f"u{i}") for i in range(3)]
        searched: list[tuple[str, int]] = []
        saved: list[str] = []

        def fake_search(search_term, page=1, **_legacy):
            searched.append((search_term, page))
            return list(pool)

        def fake_save(video_url, save_dir=""):
            saved.append(video_url)
            return f"/saved/{video_url}"

        def fake_judge(item, segment_text="", search_term=""):
            return {"verdict": "relevant", "reason": "ok", "image_source": "poster",
                    "asset_id": item.source_info.get("asset_id", "stub")}

        bombs = [
            patch.object(image_embedding, "embed_text", side_effect=AssertionError("no query embed")),
            patch.object(image_embedding, "embed_image", side_effect=AssertionError("no candidate embed")),
            patch.object(video_match, "fetch_image_bytes", side_effect=AssertionError("no preview fetch")),
            patch.object(video_match.llm, "generate_response", side_effect=AssertionError("no llm")),
            patch.object(video_match, "generate_segment_queries", return_value=queries),
            patch.object(video_match.material_rerank, "is_rerank_enabled", return_value=False),
            patch.object(video_match.material_rerank, "_walk_limit", return_value=10),
        ]
        for bomb in bombs:
            bomb.start()
            self.addCleanup(bomb.stop)

        results = video_match.match_segments(
            segments=[{"index": 0, "text": "one", "duration": 12}],
            video_subject="panda",
            search_videos=fake_search,
            save_video=fake_save,
            video_aspect="portrait",
            clip_duration=5,
            judge_candidate=fake_judge,
            embedding_gate=None,
            generate_image=lambda segment, duration=None, refined_prompt=None, framing="": (
                "/saved/gen.mp4",
                {"source": "stub"},
            ),
        )

        # 整句先于角度词条进池（ADR-0019 决策 1）。
        self.assertEqual(searched[0][0], "A broad scene.")
        self.assertIn(("angle one", 1), searched)
        self.assertIn(("angle two", 1), searched)
        # 池序即库序：重排关闭时按库序采纳（无 shuffle、无嵌入排序）。
        self.assertEqual(saved, ["u0", "u1", "u2"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].clips, ["/saved/u0", "/saved/u1", "/saved/u2"])


class TestEmbedViaServiceTimeout(unittest.TestCase):
    """`embed_via_service` 的 [library] timeout 兜底：非法配置不穿透、不停用。"""

    def setUp(self):
        self._library_cfg = dict(config.library)
        response = type(
            "R", (), {"status_code": 200, "json": lambda self: {"vector": [0.1, 0.2]}}
        )
        patcher = patch.object(image_embedding.requests, "post", return_value=response())
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        config.library.clear()
        config.library.update(self._library_cfg)

    def test_timeout_reads_library_section(self):
        config.library["timeout"] = 5
        self.assertEqual(image_embedding.embed_via_service("pexels:1"), [0.1, 0.2])
        self.assertEqual(
            image_embedding.requests.post.call_args.kwargs["timeout"], (5.0, 5.0)
        )

    def test_bad_timeout_config_falls_back_to_30s(self):
        # 与 material._library_settings 同款兜底：非法值回落 30s，配置错误
        # 既不抛 ValueError 穿透 fail-open 契约，也不静默停用向量路径。
        config.library["timeout"] = "30s"
        self.assertEqual(image_embedding.embed_via_service("pexels:2"), [0.1, 0.2])
        self.assertEqual(
            image_embedding.requests.post.call_args.kwargs["timeout"], (30.0, 30.0)
        )


if __name__ == "__main__":
    unittest.main()
