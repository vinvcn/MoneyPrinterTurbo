"""
task.py 查重门接线测试。

只测接线契约：duplicate_gate 开启时按任务建一个 EmbeddingGate 并注入
任务级 vector_cache（与 video_match 粗排共享同一份 url->向量缓存），
随 match_segments 一起下发；关闭时不建门、gate/judge 传 None。
VLM judge 的注入行为归 T3（test_vlm_judge.py），本文件不覆盖。
material_rerank 的启用状态不影响门的构造。全部 mock，无网络请求。
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoParams
from app.services import task


class TestTaskGateWiring(unittest.TestCase):
    def _run_pipeline_to_materials(self, duplicate_enabled):
        """跑 stop_at="materials" 流水线，返回素材层收到的 kwargs 与 gate mock。"""
        captured = {}
        gate = MagicMock()
        gate.register_accepted = MagicMock()

        def fake_match(**kwargs):
            captured.update(kwargs)
            return [SimpleNamespace(clips=["clip.mp4"])]

        gate_factory = MagicMock(return_value=gate)
        patchers = [
            patch.object(task.sm.state, "update_task", lambda *a, **k: None),
            patch.object(
                task.segmenter,
                "segment_script",
                lambda script: [SimpleNamespace(index=0, text="hello world")],
            ),
            patch.object(task, "save_script_data", lambda *a, **k: None),
            patch.object(
                task.segment_material,
                "english_search_term",
                lambda text: "noodles" if text else "",
            ),
            patch.object(
                task.segment_audio,
                "prepare_segment_audio",
                lambda **kwargs: SimpleNamespace(
                    ok=True,
                    audio_file="a.mp3",
                    total_duration_ms=1500,
                    error=None,
                ),
            ),
            patch.object(
                task.image_embedding,
                "is_duplicate_gate_enabled",
                return_value=duplicate_enabled,
            ),
            patch.object(
                task.image_embedding,
                "make_default_gate",
                new=gate_factory,
            ),
            patch.object(task.vlm_judge, "is_enabled", return_value=False),
            patch.object(
                task.material,
                "search_videos_with_cache_for_source",
                lambda source, page=1: lambda *a, **k: [],
            ),
            patch.object(task.video_match, "match_segments", fake_match),
            patch.object(
                task.segment_material,
                "persist_segment_material_sources",
                lambda *a, **k: None,
            ),
            patch.object(
                task.segment_material,
                "segments_to_records",
                lambda materials: [],
            ),
            patch.object(task.utils, "task_dir", lambda sub_dir="": "/tmp"),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        params = VideoParams(
            video_subject="hello",
            subtitle_enabled=False,
            video_clip_duration=3,
        )
        task._run_segment_first_pipeline(
            "task-wiring", params, "hello world", stop_at="materials"
        )
        return captured, gate_factory, gate

    def test_gate_built_with_vector_cache_when_duplicate_gate_enabled(self):
        captured, gate_factory, gate = self._run_pipeline_to_materials(True)
        gate_factory.assert_called_once()
        # 门按任务注入任务级向量缓存：match_segments 取回同一份 dict 供
        # 粗排预热，粗排与查重门共享向量（同一 URL 全链路只嵌入一次）。
        self.assertIsInstance(
            gate_factory.call_args.kwargs.get("vector_cache"), dict
        )
        self.assertIs(captured["embedding_gate"], gate)
        self.assertIsNone(captured["judge_candidate"])
        self.assertTrue(callable(captured["generate_image"]))
        self.assertTrue(callable(captured["search_videos"]))
        self.assertTrue(callable(captured["save_video"]))

    def test_no_gate_when_duplicate_gate_disabled(self):
        captured, gate_factory, _ = self._run_pipeline_to_materials(False)
        gate_factory.assert_not_called()
        self.assertIsNone(captured["embedding_gate"])
        self.assertIsNone(captured["judge_candidate"])

    def test_rerank_irrelevant_to_gate_construction(self):
        """material_rerank 启用不触发门构造；门只由 duplicate_gate 决定。"""
        with patch(
            "app.services.material_rerank.is_rerank_enabled", return_value=True
        ):
            captured, gate_factory, _ = self._run_pipeline_to_materials(False)
            gate_factory.assert_not_called()
            self.assertIsNone(captured["embedding_gate"])
            self.assertIsNone(captured["judge_candidate"])


if __name__ == "__main__":
    unittest.main()
