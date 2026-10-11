"""Regression tests for benchmark graders and dataset aliases."""

from __future__ import annotations

import json

from benchmark.datasets import load_triviaqa
from benchmark.graders import _extract_number, grade_gsm8k, grade_triviaqa


class TestTriviaQAGrader:
    def test_number_is_not_matched_as_substring(self):
        assert not grade_triviaqa("The answer is 120", "12")
        assert grade_triviaqa("The answer is 12.", "12")

    def test_word_is_not_matched_inside_another_word(self):
        assert not grade_triviaqa("Parisian cafes", "Paris")

    def test_aliases_are_accepted(self):
        assert grade_triviaqa("It was JFK.", ["John F. Kennedy", "JFK"])
        assert not grade_triviaqa("It was Nixon.", ["John F. Kennedy", "JFK"])

    def test_empty_gold_never_matches(self):
        assert not grade_triviaqa("anything", "")
        assert not grade_triviaqa("anything", [])


class TestGSM8KGrader:
    def test_lone_comma_is_not_a_number(self):
        assert _extract_number("a, b") is None
        assert _extract_number("costs 5, then more, ") == "5"

    def test_negative_and_decimal(self):
        assert grade_gsm8k("so the result is -3.5", "-3.5")

    def test_hash_format_preferred_over_last_number(self):
        assert grade_gsm8k("#### 42\nchecked against 7 cases", "42")


class TestFinalAnswerLine:
    def test_only_final_answer_line_is_graded_for_triviaqa(self):
        text = "Some say Paris, but no.\nFinal answer: Lyon"
        assert grade_triviaqa(text, "Lyon")
        assert not grade_triviaqa(text, "Paris")

    def test_without_marker_whole_text_is_graded(self):
        assert grade_triviaqa("I think it is Paris.", "Paris")

    def test_gsm8k_uses_first_number_on_final_answer_line(self):
        text = "5 * 2 = 10, 10 + 2 = 12\nFinal answer: 18 vacuum cleaners in 3 houses"
        assert grade_gsm8k(text, "18")

    def test_gsm8k_marker_is_case_insensitive_and_handles_dollars(self):
        assert grade_gsm8k("so...\nFINAL ANSWER: $1,200", "1200")

    def test_last_marker_wins(self):
        assert grade_gsm8k("Final answer: 3\nWait, recheck.\nFinal answer: 4", "4")


class TestTriviaQALoaderAliases:
    def test_loader_keeps_aliases(self, tmp_path):
        path = tmp_path / "tqa.json"
        path.write_text(json.dumps([
            {"question": "Q?", "answer": {"value": "John F. Kennedy", "aliases": ["JFK", "John F. Kennedy"]}}
        ]), encoding="utf-8")
        (_, _, gold), = load_triviaqa(str(path))
        assert gold == ["John F. Kennedy", "JFK"]
        assert grade_triviaqa("JFK", gold)
