"""
Tests for _parse_classification and the cross-run signal lifecycle.

_parse_classification(reaction, text) → {effect, trend, ratio, persistent}
  - Must reject template echoes (lines containing '[')
  - Must parse DRUG_SPECIFIC / CLASS_EFFECT correctly
  - Must parse ratio robustly: "5.36x", "5.36 x", "drug is 7x"
  - Must set persistent flag when PERSISTENT appears

Signal lifecycle (classify_signals logic):
  NEW        — reaction not in prior run
  VALIDATED  — in prior run AND prr >= 0.5 × prior_prr
  DISMISSED  — in prior run AND prr < 0.5 × prior_prr
"""
import pytest
from agent.pipeline import _parse_classification


class TestParseClassification:

    def test_drug_specific_parsed(self):
        text = (
            "get_prr returned: {}\n"
            "CLASSIFICATION: DRUG_SPECIFIC | GROWING\n"
            "RATIO: WARFARIN is 8.5x lowest comparator (APIXABAN=1.2)\n"
            "INSIGHT: strong drug-specific signal"
        )
        out = _parse_classification("WARFARIN", text)
        assert out["effect"] == "DRUG_SPECIFIC"
        assert out["trend"]  == "GROWING"
        assert out["ratio"]  == pytest.approx(8.5)
        assert out["persistent"] is False

    def test_class_effect_parsed(self):
        text = "CLASSIFICATION: CLASS_EFFECT | STABLE\nRATIO: DRUG is 2.1x lowest\n"
        out = _parse_classification("NAUSEA", text)
        assert out["effect"] == "CLASS_EFFECT"
        assert out["trend"]  == "STABLE"
        assert out["ratio"]  == pytest.approx(2.1)

    def test_persistent_flag(self):
        text = "CLASSIFICATION: DRUG_SPECIFIC | GROWING | PERSISTENT\n"
        out = _parse_classification("PANCREATITIS", text)
        assert out["persistent"] is True

    def test_template_echo_rejected(self):
        """Model parroting the prompt template (contains '[') should not match."""
        text = "CLASSIFICATION: [CLASS_EFFECT|DRUG_SPECIFIC] | [GROWING|STABLE|EMERGING]\n"
        out = _parse_classification("REACTION_X", text)
        assert out["effect"] is None
        assert out["trend"]  is None

    def test_ratio_with_space_before_x(self):
        """'5.36 x' (space before x) should parse correctly."""
        text = "CLASSIFICATION: DRUG_SPECIFIC | STABLE\nRATIO: DRUG is 5.36 x lowest\n"
        out = _parse_classification("DRUG", text)
        assert out["ratio"] == pytest.approx(5.36)

    def test_ratio_integer(self):
        """Integer ratios like '7x' should parse."""
        text = "CLASSIFICATION: DRUG_SPECIFIC | STABLE\nRATIO: DRUG is 7x lowest\n"
        out = _parse_classification("DRUG", text)
        assert out["ratio"] == pytest.approx(7.0)

    def test_no_classification_line(self):
        """Missing CLASSIFICATION line → all None, no crash."""
        text = "INSIGHT: something happened"
        out = _parse_classification("REACTION", text)
        assert out["effect"] is None
        assert out["ratio"]  is None

    def test_no_ratio_line(self):
        """Missing RATIO line → ratio=None, effect still parsed."""
        text = "CLASSIFICATION: CLASS_EFFECT | STABLE\n"
        out = _parse_classification("REACTION", text)
        assert out["effect"] == "CLASS_EFFECT"
        assert out["ratio"]  is None

    def test_case_insensitive(self):
        """Parser should be case-insensitive."""
        text = "classification: drug_specific | growing\nratio: drug is 6.0x\n"
        out = _parse_classification("R", text)
        assert out["effect"] == "DRUG_SPECIFIC"
        assert out["ratio"]  == pytest.approx(6.0)


class TestPhase2Trigger:
    """The interesting flag uses parser output, not fragile text scraping."""

    def _is_interesting(self, classifications):
        return any(
            c["effect"] == "DRUG_SPECIFIC"
            or (c["ratio"] is not None and c["ratio"] > 5.0)
            for c in classifications
        )

    def test_drug_specific_fires_phase2(self):
        cls = [{"effect": "DRUG_SPECIFIC", "ratio": 3.0, "trend": "STABLE"}]
        assert self._is_interesting(cls) is True

    def test_high_ratio_fires_phase2(self):
        cls = [{"effect": "CLASS_EFFECT", "ratio": 6.5, "trend": "STABLE"}]
        assert self._is_interesting(cls) is True

    def test_threshold_is_5_not_7(self):
        """Ratio 5.01 must trigger Phase 2 (old threshold was > 7, dead band 5-7)."""
        cls = [{"effect": "CLASS_EFFECT", "ratio": 5.01, "trend": "STABLE"}]
        assert self._is_interesting(cls) is True

    def test_low_ratio_class_effect_no_phase2(self):
        cls = [{"effect": "CLASS_EFFECT", "ratio": 2.5, "trend": "STABLE"}]
        assert self._is_interesting(cls) is False

    def test_empty_classifications_no_phase2(self):
        assert self._is_interesting([]) is False

    def test_none_ratio_handled(self):
        """ratio=None should not crash the trigger."""
        cls = [{"effect": "CLASS_EFFECT", "ratio": None, "trend": "STABLE"}]
        assert self._is_interesting(cls) is False


class TestSignalLifecycle:
    """NEW / VALIDATED / DISMISSED assignment logic."""

    def _classify(self, current_prr, prior_prr=None):
        if prior_prr is None:
            return "NEW"
        return "VALIDATED" if current_prr >= prior_prr * 0.5 else "DISMISSED"

    def test_new_signal(self):
        assert self._classify(8.5, prior_prr=None) == "NEW"

    def test_validated_same_prr(self):
        assert self._classify(8.5, prior_prr=8.5) == "VALIDATED"

    def test_validated_slightly_lower(self):
        """PRR dropped to 60% of prior — still VALIDATED (above 50% threshold)."""
        assert self._classify(5.1, prior_prr=8.5) == "VALIDATED"

    def test_dismissed_collapsed(self):
        """PRR collapsed to 20% of prior — DISMISSED."""
        assert self._classify(1.7, prior_prr=8.5) == "DISMISSED"

    def test_threshold_boundary(self):
        """Exactly at 50% → VALIDATED (boundary inclusive)."""
        assert self._classify(4.25, prior_prr=8.5) == "VALIDATED"

    def test_below_threshold(self):
        """49% → DISMISSED."""
        assert self._classify(4.1, prior_prr=8.5) == "DISMISSED"
