import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import task_artifacts


class TestTaskArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.task_dir = Path(self.temp_dir.name)
        self.task_dir_patch = patch(
            "app.services.task_artifacts.utils.task_dir",
            return_value=str(self.task_dir),
        )
        self.task_dir_patch.start()

    def tearDown(self):
        self.task_dir_patch.stop()
        self.temp_dir.cleanup()

    def test_patch_preserves_existing_script_fields(self):
        """补充素材来源时不能覆盖历史任务恢复依赖的文案、关键词和参数。"""
        original = {
            "script": "existing script",
            "search_terms": ["nature"],
            "params": {"video_source": "pixabay"},
        }
        task_artifacts.write_script_data("task-1", original)

        updated = task_artifacts.patch_script_data(
            "task-1",
            material_sources=[
                {
                    "provider": "pixabay",
                    "asset_id": "123",
                    "local_file": "vid-123.mp4",
                }
            ],
        )
        payload = json.loads((self.task_dir / "script.json").read_text())

        self.assertTrue(updated)
        self.assertEqual(payload["script"], original["script"])
        self.assertEqual(payload["search_terms"], original["search_terms"])
        self.assertEqual(payload["params"], original["params"])
        self.assertEqual(payload["material_sources"][0]["asset_id"], "123")
        self.assertEqual(list(self.task_dir.glob(".script.json.*.tmp")), [])

    def test_write_script_data_serializes_video_params(self):
        """原子写入替换旧实现后，仍需完整兼容任务主流程传入的 Pydantic 参数。"""
        params = VideoParams(
            video_subject="test subject",
            video_terms=["city", "night"],
        )

        task_artifacts.write_script_data(
            "task-params",
            {
                "script": "test script",
                "search_terms": ["city"],
                "params": params,
            },
        )
        payload = json.loads((self.task_dir / "script.json").read_text())

        self.assertEqual(payload["params"]["video_subject"], "test subject")
        self.assertEqual(payload["params"]["video_terms"], ["city", "night"])
        self.assertEqual(payload["params"]["video_source"], "pexels")

    def test_patch_missing_script_is_non_blocking(self):
        """独立调用素材下载时没有任务清单，应静默跳过而不是创建残缺 JSON。"""
        updated = task_artifacts.patch_script_data(
            "standalone",
            material_sources=[],
        )

        self.assertFalse(updated)
        self.assertFalse((self.task_dir / "script.json").exists())

    def test_patch_invalid_script_returns_false_without_overwrite(self):
        """历史 JSON 损坏时必须保留原文件、记录错误，并允许视频主流程继续。"""
        target = self.task_dir / "script.json"
        target.write_text("{invalid-json", encoding="utf-8")

        with patch.object(task_artifacts.logger, "warning") as warning:
            updated = task_artifacts.patch_script_data(
                "task-1",
                material_sources=[],
            )

        self.assertFalse(updated)
        self.assertEqual(target.read_text(encoding="utf-8"), "{invalid-json")
        self.assertTrue(warning.called)

    def test_read_segment_materials_returns_persisted_records(self):
        """persist_segment_material_sources 补写的每段溯源必须原样读回。"""
        task_artifacts.write_script_data("task-seg", {"script": "s"})
        task_artifacts.patch_script_data(
            "task-seg",
            segment_materials=[
                {
                    "index": 0,
                    "clip_sources": [
                        {"url": "/shared/cache/m/f/clip.mp4", "local_file": "clip.mp4"}
                    ],
                    "fallback_level": "",
                }
            ],
        )

        records = task_artifacts.read_segment_materials("task-seg")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["index"], 0)
        self.assertEqual(
            records[0]["clip_sources"][0]["url"],
            "/shared/cache/m/f/clip.mp4",
        )

    def test_read_segment_materials_missing_manifest_is_empty(self):
        """任务还没落清单时读取是空列表，不是异常。"""
        self.assertEqual(task_artifacts.read_segment_materials("never-written"), [])

    def test_read_segment_materials_absent_field_is_empty(self):
        """早于溯源字段的历史清单没有该键，同样返回空列表。"""
        task_artifacts.write_script_data("task-old", {"script": "legacy"})

        self.assertEqual(task_artifacts.read_segment_materials("task-old"), [])

    def test_read_segment_materials_invalid_payload_degrades(self):
        """清单损坏或字段类型不符时降级为空列表，绝不让任务查询失败。"""
        task_artifacts.write_script_data("task-bad", {"script": "s"})
        task_artifacts.patch_script_data(
            "task-bad", segment_materials="not-a-list"
        )
        broken = self.task_dir / "script.json"
        self.assertEqual(task_artifacts.read_segment_materials("task-bad"), [])

        broken.write_text("{invalid-json", encoding="utf-8")
        with patch.object(task_artifacts.logger, "warning") as warning:
            self.assertEqual(task_artifacts.read_segment_materials("task-bad"), [])
        # 读取失败只告警不抛；原文件不被改写。
        self.assertEqual(broken.read_text(encoding="utf-8"), "{invalid-json")
        self.assertTrue(warning.called)


if __name__ == "__main__":
    unittest.main()
