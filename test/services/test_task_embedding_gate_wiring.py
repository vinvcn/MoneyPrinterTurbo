"""
task.py 查重门/粗筛接线测试（embed-prefilter T4）。

只测接线契约：duplicate_gate 与 coarse_filter 任一开启时按任务建一个
EmbeddingGate 并把 register_accepted 注入素材层；两者都关闭时不建门、
不注入回调。VLM judge 的注入行为归 T3（test_vlm_judge.py /
test_segment_material_quota.py），本文件不覆盖。全部 mock，无网络请求。
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
    def _run_pipeline_to_materials(self, duplicate_enabled, coarse_enabled):
        """跑 stop_at="materials" 流水线，返回素材层收到的 kwargs 与 gate mock。"""
        captured = {}
        gate = MagicMock()

        def fake_prepare(**kwargs):
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
            patch.object(
                task.segment_terms,
                "extract_terms_for_segments",
                lambda records, video_subject: {},
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
                "is_coarse_filter_enabled",
                return_value=coarse_enabled,
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
                lambda source: lambda *a, **k: [],
            ),
            patch.object(
                task.segment_material, "prepare_segment_materials", fake_prepare
            ),
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

    def test_gate_built_when_duplicate_gate_enabled(self):
        captured, gate_factory, gate = self._run_pipeline_to_materials(True, False)
        gate_factory.assert_called_once()
        self.assertIs(captured["on_clip_accepted"], gate.register_accepted)

    def test_gate_built_when_only_coarse_filter_enabled(self):
        captured, gate_factory, gate = self._run_pipeline_to_materials(False, True)
        gate_factory.assert_called_once()
        self.assertIs(captured["on_clip_accepted"], gate.register_accepted)

    def test_no_gate_when_both_flags_off(self):
        captured, gate_factory, _ = self._run_pipeline_to_materials(False, False)
        gate_factory.assert_not_called()
        self.assertIsNone(captured["on_clip_accepted"])
        self.assertIsNone(captured["judge_candidate"])


if __name__ == "__main__":
    unittest.main()
