"""
Tests for the sentence-aware, negation-aware, direction-aware label matcher.

_is_labeled(reaction: str, label_text: str) -> bool

All tests use simplified label text strings — no OpenSearch required.
"""
import pytest
from agent.pipeline import _is_labeled


class TestBasicMatching:
    def test_exact_match(self):
        """Straightforward match when words appear in the label."""
        assert _is_labeled("PANCREATITIS", "Pancreatitis has been reported.") is True

    def test_word_order_independent(self):
        """'PANCREATITIS ACUTE' should match 'acute pancreatitis' in label."""
        assert _is_labeled("PANCREATITIS ACUTE", "Acute pancreatitis was observed.") is True

    def test_unrelated_reaction_not_matched(self):
        """Reaction absent from label → False."""
        assert _is_labeled("PANCREATITIS", "No gastrointestinal events reported.") is False


class TestNegationAwareness:
    def test_simple_negation(self):
        """'no evidence of pancreatitis' → False."""
        assert _is_labeled("PANCREATITIS", "no evidence of pancreatitis was found") is False

    def test_negative_then_positive_cross_sentence(self):
        """
        Cross-sentence safety: negation in sentence 1 must NOT suppress
        a genuine match in sentence 2.
        """
        label = "no evidence of pancreatitis in early trials. pancreatitis occurred in 3% of patients."
        assert _is_labeled("PANCREATITIS", label) is True

    def test_without_negation(self):
        """'without pancreatitis' → False."""
        assert _is_labeled("PANCREATITIS", "patients without pancreatitis were enrolled") is False


class TestDirectionAwareness:
    def test_blood_glucose_increased_matched(self):
        """'BLOOD GLUCOSE INCREASED' matched by upward-direction label text."""
        label = "blood glucose was elevated in some patients"
        assert _is_labeled("BLOOD GLUCOSE INCREASED", label) is True

    def test_blood_glucose_decreased_not_matched_by_increased_label(self):
        """
        'BLOOD GLUCOSE DECREASED' must NOT match a label that only says
        'increased' — critical safety requirement.
        """
        label = "blood glucose increased in some patients receiving treatment"
        assert _is_labeled("BLOOD GLUCOSE DECREASED", label) is False

    def test_blood_glucose_decreased_matched_correctly(self):
        """'BLOOD GLUCOSE DECREASED' matched when label explicitly says 'decreased'."""
        label = "blood glucose decreased and hypoglycemia was reported"
        assert _is_labeled("BLOOD GLUCOSE DECREASED", label) is True


class TestSynonymExpansion:
    def test_impaired_gastric_emptying_via_delay_synonym(self):
        """
        'IMPAIRED GASTRIC EMPTYING' should match label text 'delays gastric emptying'
        because impaired↔delays is a registered synonym pair.
        """
        label = "semaglutide delays gastric emptying in patients"
        assert _is_labeled("IMPAIRED GASTRIC EMPTYING", label) is True

    def test_haemorrhage_us_spelling(self):
        """haemorrhage ↔ hemorrhage synonym: UK spelling in PT matches US spelling in label."""
        label = "gastrointestinal hemorrhage has been reported"
        assert _is_labeled("GASTROINTESTINAL HAEMORRHAGE", label) is True


class TestEdgeCases:
    def test_empty_label_returns_false(self):
        assert _is_labeled("PANCREATITIS", "") is False

    def test_reaction_with_stop_words_only(self):
        """A reaction whose tokens are all stop words → falls back to substring match."""
        # 'DISEASE NOS' — 'nos' and 'disease' are both stop words
        # Expected: False (substring not in label)
        assert _is_labeled("DISEASE NOS", "no relevant findings") is False
