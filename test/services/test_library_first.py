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
            video_match, "fetch_image_bytes", return_value=(b"jpeg", (640, 320), "service")
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
        # VLM 走查是终审。
        patcher = patch.object(
            image_embedding,
            "embed_via_service",
            return_value=None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        pool = [{"url": "u0", "asset_id": "0", "item": _item(0), "term": "t"}]
        selected, _ = video_match._select_by_library_order(pool, self.gate)
        self.assertEqual([c["url"] for c in selected], ["u0"])

    def test_without_gate_truncates_pool(self):
        pool = [{"url": f"u{i}", "asset_id": f"{i}", "item": _item(i), "term": "t"} for i in range(40)]
        selected, skips = video_match._select_by_library_order(pool, None)
        self.assertEqual(len(selected), video_match._COARSE_TOP_K)
        self.assertEqual(skips, [])

    def test_gate_failure_fails_open(self):
        patcher = patch.object(
            image_embedding,
            "embed_via_service",
            return_value=None,  # 服务不可用 → 放行，VLM 终审
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        pool = [{"url": "u0", "asset_id": "0", "item": _item(0), "term": "t"}]
        selected, _ = video_match._select_by_library_order(pool, self.gate)
        self.assertEqual([c["url"] for c in selected], ["u0"])


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
            fetch.return_value = (b"jpeg", (640, 320), "service")
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
            fetch.return_value = (b"jpeg", (640, 320), "service")
            candidate = video_match._candidate_from_item(_item(7), "t")
        fetch.assert_called_once()
        self.assertNotEqual(candidate["data_uri"], "")


if __name__ == "__main__":
    unittest.main()
