import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import segmenter


class TestSplitNarrationUnits(unittest.TestCase):
    def test_empty_script_returns_no_units(self):
        self.assertEqual(segmenter.split_narration_units(""), [])
        self.assertEqual(segmenter.split_narration_units("   \n  "), [])

    def test_english_sentences_split_on_punctuation(self):
        text = "The sun rises in the east. It sets in the west! Is that surprising?"
        units = segmenter.split_narration_units(text)
        self.assertEqual(
            units,
            [
                "The sun rises in the east.",
                "It sets in the west!",
                "Is that surprising?",
            ],
        )

    def test_chinese_sentences_split_on_punctuation(self):
        text = "春天的花海如诗如画。大地复苏，万物生长。"
        units = segmenter.split_narration_units(text)
        self.assertEqual(units, ["春天的花海如诗如画。", "大地复苏，万物生长。"])

    def test_newline_is_a_hard_boundary(self):
        text = "First line without ending punctuation\nSecond line."
        units = segmenter.split_narration_units(text)
        self.assertEqual(units, ["First line without ending punctuation", "Second line."])

    def test_decimal_numbers_are_not_split(self):
        text = "The fee is 2.5 percent. It was charged at 1,000 units."
        units = segmenter.split_narration_units(text)
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0], "The fee is 2.5 percent.")

    def test_text_without_punctuation_stays_one_unit(self):
        text = "just a phrase drifting without end"
        self.assertEqual(segmenter.split_narration_units(text), [text])


class TestEstimateDuration(unittest.TestCase):
    def test_empty_text_has_zero_duration(self):
        self.assertEqual(segmenter.estimate_duration_seconds(""), 0.0)

    def test_english_estimated_from_word_rate(self):
        # 27 words at 2.7 words/second = 10 seconds.
        text = " ".join(["word"] * 27)
        self.assertAlmostEqual(segmenter.estimate_duration_seconds(text), 10.0, places=6)

    def test_chinese_estimated_from_character_rate(self):
        # 42 CJK characters at 4.2 chars/second = 10 seconds.
        text = "字" * 42
        self.assertAlmostEqual(segmenter.estimate_duration_seconds(text), 10.0, places=6)


class TestSegmentScript(unittest.TestCase):
    def test_single_short_sentence_is_one_segment(self):
        segments = segmenter.segment_script("Hello world.")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "Hello world.")
        self.assertEqual(segments[0].index, 0)
        self.assertGreater(segments[0].estimated_duration, 0)

    def test_empty_script_returns_no_segments(self):
        self.assertEqual(segmenter.segment_script("   "), [])

    def test_segments_merge_below_minimum_duration(self):
        text = "One. Two. Three. Four. Five."
        segments = segmenter.segment_script(text, min_duration=6.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "One. Two. Three. Four. Five.")

    def test_long_narration_splits_into_multiple_segments(self):
        text = " ".join(["word"] * 27) + ". " + " ".join(["next"] * 27) + "."
        segments = segmenter.segment_script(text, min_duration=0.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, (" ".join(["word"] * 27) + ".").strip())
        self.assertEqual(segments[1].text, (" ".join(["next"] * 27) + ".").strip())

    def test_segment_indexes_are_sequential(self):
        text = "\n".join("Sentence number %d." % i for i in range(1, 9))
        segments = segmenter.segment_script(text, min_duration=0.0)
        self.assertEqual([s.index for s in segments], list(range(len(segments))))

    def test_segments_preserve_newline_separators_between_units(self):
        text = "First paragraph here.\nSecond paragraph here."
        segments = segmenter.segment_script(text, min_duration=0.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "First paragraph here.")
        self.assertEqual(segments[1].text, "Second paragraph here.")

    def test_max_duration_is_respected(self):
        # 60 words at 2.7 words/second ≈ 22s; max 10s must force a split.
        text = " ".join(["word"] * 60) + "."
        segments = segmenter.segment_script(text, min_duration=0.0, max_duration=10.0)
        self.assertGreater(len(segments), 1)
        for segment in segments:
            self.assertLessEqual(segment.estimated_duration, 10.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
