import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import segment_material as sm


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
        """CJK 过滤规则：含中日韩字符或空白词条返回空串，纯英文原样返回。"""
        self.assertEqual(sm.english_search_term("city skyline"), "city skyline")
        self.assertEqual(sm.english_search_term("  city skyline  "), "city skyline")
        self.assertEqual(sm.english_search_term("city skyline 城市"), "")
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


if __name__ == "__main__":
    unittest.main()
