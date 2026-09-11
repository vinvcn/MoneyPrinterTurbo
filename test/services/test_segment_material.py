import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import segment_material as sm
from app.services.segment_material import MAX_SEARCH_PAGES, _search_pages


class TestRecordsConversion(unittest.TestCase):
    def test_segments_to_records_is_json_safe(self):
        materials = [
            sm.SegmentMaterials(
                index=0,
                search_term="a",
                clips=["/x/a1.mp4"],
                resolved_term="a",
                fallback_level="self",
            )
        ]
        records = sm.segments_to_records(materials)
        self.assertEqual(records[0]["clips"], ["/x/a1.mp4"])
        self.assertEqual(records[0]["index"], 0)

    def test_segment_materials_defaults_are_independent(self):
        """SegmentMaterials 的可变默认字段逐实例独立（dataclass field 语义）。"""
        first = sm.SegmentMaterials(index=0, search_term="a")
        second = sm.SegmentMaterials(index=1, search_term="b")
        first.clips.append("/x/a.mp4")
        first.search_attempts.append({"level": "self", "term": "a", "found": True})
        self.assertEqual(second.clips, [])
        self.assertEqual(second.search_attempts, [])

    def test_english_search_term_filters_cjk(self):
        """CJK 过滤规则：剔除 CJK 字符保留英文部分，纯 CJK 或空白返回空串。"""
        self.assertEqual(sm.english_search_term("city skyline"), "city skyline")
        self.assertEqual(sm.english_search_term("  city skyline  "), "city skyline")
        self.assertEqual(sm.english_search_term("city skyline 城市"), "city skyline")
        self.assertEqual(sm.english_search_term("大熊猫 panda daily"), "panda daily")
        self.assertEqual(sm.english_search_term("panda大熊猫4k"), "panda 4k")
        self.assertEqual(sm.english_search_term("这是中文"), "")
        self.assertEqual(sm.english_search_term(""), "")
        self.assertEqual(sm.english_search_term(None), "")


class TestPersistSegmentMaterialSources(unittest.TestCase):
    def test_persist_writes_records_and_survives_failure(self):
        """清单落盘是 best-effort：正常时写入逐段记录，异常只降级为告警。"""
        materials = [
            sm.SegmentMaterials(
                index=0,
                search_term="a",
                clips=["/x/a1.mp4"],
                resolved_term="a",
                fallback_level="self",
            )
        ]
        with patch.object(
            sm.task_artifacts, "patch_script_data", return_value=True
        ) as persist:
            sm.persist_segment_material_sources("task-1", materials)
        self.assertEqual(persist.call_args.args[0], "task-1")
        record = persist.call_args.kwargs["segment_materials"][0]
        self.assertEqual(record["clips"], ["a1.mp4"])

        with patch.object(
            sm.task_artifacts,
            "patch_script_data",
            side_effect=RuntimeError("disk full"),
        ):
            # 不抛异常：清单是辅助记录，绝不中断视频生成。
            sm.persist_segment_material_sources("task-1", materials)


class TestSearchPages(unittest.TestCase):
    """_search_pages 读取语义：键缺失、非法或小于 1 时回落 MAX_SEARCH_PAGES
    （镜像 material_rerank._walk_limit 的 fail-open 先例）。全部用例通过
    patch.dict 隔离 live config——开发机 config.toml 后续加入
    max_search_pages = 1 也不会翻转任何断言。"""

    def test_absent_key_falls_back_to_constant(self):
        """键缺失 → 回落 MAX_SEARCH_PAGES；clear=True 构造"无键"字典，
        不依赖 live config 的当前状态（用户后续加 max_search_pages = 1
        也不会翻转本测试）。"""
        with patch.dict(config.material_rerank, {}, clear=True):
            self.assertEqual(_search_pages(), MAX_SEARCH_PAGES)

    def test_valid_value_one_is_honored(self):
        with patch.dict(config.material_rerank, {"max_search_pages": 1}):
            self.assertEqual(_search_pages(), 1)

    def test_zero_falls_back_to_default(self):
        with patch.dict(config.material_rerank, {"max_search_pages": 0}):
            self.assertEqual(_search_pages(), MAX_SEARCH_PAGES)

    def test_non_numeric_falls_back_to_default(self):
        with patch.dict(config.material_rerank, {"max_search_pages": "abc"}):
            self.assertEqual(_search_pages(), MAX_SEARCH_PAGES)

    def test_valid_value_three_is_honored(self):
        with patch.dict(config.material_rerank, {"max_search_pages": 3}):
            self.assertEqual(_search_pages(), 3)


class TestSegmentsToRecordsHoles(unittest.TestCase):
    """Roundtrip test for the holes field added to SegmentMaterials."""

    def test_holes_with_value(self):
        """SegmentMaterials with holes=[2] serializes 'holes': [2]."""
        materials = [
            sm.SegmentMaterials(
                index=0,
                search_term="a",
                resolved_term="a",
                fallback_level="self",
                clips=["/x/a1.mp4"],
                holes=[2],
            )
        ]
        records = sm.segments_to_records(materials)
        self.assertIn("holes", records[0])
        self.assertEqual(records[0]["holes"], [2])

    def test_holes_default_empty(self):
        """Default-constructed SegmentMaterials serializes 'holes': []."""
        materials = [
            sm.SegmentMaterials(
                index=0,
                search_term="a",
                resolved_term="a",
                fallback_level="self",
                clips=["/x/a1.mp4"],
            )
        ]
        records = sm.segments_to_records(materials)
        self.assertIn("holes", records[0])
        self.assertEqual(records[0]["holes"], [])

    def test_backward_compat_no_holes_kwarg(self):
        """Constructing without holes works (backward compat)."""
        m = sm.SegmentMaterials(index=0, search_term="x")
        self.assertEqual(m.holes, [])
        records = sm.segments_to_records([m])
        self.assertEqual(records[0]["holes"], [])


if __name__ == "__main__":
    unittest.main()
