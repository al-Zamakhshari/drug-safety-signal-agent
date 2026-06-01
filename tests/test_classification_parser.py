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
        """Alternation template [X|Y|Z] (pipe inside brackets) → rejected."""
        text = "CLASSIFICATION: [CLASS_EFFECT|DRUG_SPECIFIC] | [GROWING|STABLE|EMERGING]\n"
        out = _parse_classification("REACTION_X", text)
        assert out["effect"] is None
        assert out["trend"]  is None

    def test_legitimate_brackets_not_rejected(self):
        """Real output with brackets that are NOT the alternation template: accepted."""
        text = "CLASSIFICATION: DRUG_SPECIFIC [confirmed] | GROWING\n"
        out = _parse_classification("REACTION_Y", text)
        assert out["effect"] == "DRUG_SPECIFIC"
        assert out["trend"]  == "GROWING"

    def test_drug_specific_wins_over_class_effect(self):
        """DRUG_SPECIFIC must win deterministically when both tokens present."""
        text = "CLASSIFICATION: CLASS_EFFECT not DRUG_SPECIFIC | STABLE\n"
        out = _parse_classification("REACTION_Z", text)
        # DRUG_SPECIFIC should win because it's first in the ordered tuple
        assert out["effect"] == "DRUG_SPECIFIC"

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
    """NEW / VALIDATED / DISMISSED assignment logic — CI-overlap version."""

    def _classify(self, c_prr, c_lo=None, c_up=None, prior=None):
        """Replicate classify_signals logic exactly."""
        if prior is None:
            return "NEW"
        p_prr = prior.get("prr", 0)
        p_lo  = prior.get("prr_lower")
        p_up  = prior.get("prr_upper")
        if None in (c_lo, c_up, p_lo, p_up):
            return "VALIDATED" if c_prr >= p_prr * 0.5 else "DISMISSED"
        if c_up < p_lo:
            return "DISMISSED"
        return "VALIDATED"

    def test_new_signal(self):
        assert self._classify(8.5) == "NEW"

    def test_ci_overlap_validated(self):
        """CIs overlap → VALIDATED, even if point PRR dropped."""
        # prior: PRR=8.5, CI=[6.0, 12.0]; current: PRR=5.5, CI=[4.5, 7.0]
        # CIs overlap (7.0 >= 6.0) → VALIDATED
        prior = {"prr": 8.5, "prr_lower": 6.0, "prr_upper": 12.0}
        assert self._classify(5.5, c_lo=4.5, c_up=7.0, prior=prior) == "VALIDATED"

    def test_ci_no_overlap_dismissed(self):
        """Current upper CI below prior lower CI → DISMISSED (genuine collapse)."""
        # prior: PRR=8.5, CI=[6.0, 12.0]; current: PRR=3.0, CI=[2.0, 4.5]
        # 4.5 < 6.0 → no overlap → DISMISSED
        prior = {"prr": 8.5, "prr_lower": 6.0, "prr_upper": 12.0}
        assert self._classify(3.0, c_lo=2.0, c_up=4.5, prior=prior) == "DISMISSED"

    def test_ci_absent_fallback_validated(self):
        """When CI absent, fall back to 50% point-estimate rule."""
        prior = {"prr": 8.5, "prr_lower": None, "prr_upper": None}
        assert self._classify(5.1, prior=prior) == "VALIDATED"   # 60% of prior

    def test_ci_absent_fallback_dismissed(self):
        prior = {"prr": 8.5, "prr_lower": None, "prr_upper": None}
        assert self._classify(1.7, prior=prior) == "DISMISSED"   # 20% of prior

    def test_prr_recovered_ci_overlap(self):
        """PRR went up — CIs also overlap → VALIDATED."""
        prior = {"prr": 5.0, "prr_lower": 3.5, "prr_upper": 7.0}
        assert self._classify(7.5, c_lo=5.5, c_up=10.0, prior=prior) == "VALIDATED"

    def test_sampling_noise_not_dismissed(self):
        """
        PRR 8.0 → 3.9 with overlapping CIs should be VALIDATED (noise), not DISMISSED.
        Old 50% cliff would give DISMISSED (3.9 < 4.0 = 8.0*0.5).
        """
        prior = {"prr": 8.0, "prr_lower": 5.5, "prr_upper": 11.5}
        # current CI [2.5, 5.8] — upper=5.8 >= prior_lower=5.5 → overlap → VALIDATED
        assert self._classify(3.9, c_lo=2.5, c_up=5.8, prior=prior) == "VALIDATED"

    def test_genuine_collapse_dismissed(self):
        """
        PRR 8.0 → 1.5 with non-overlapping CIs should be DISMISSED.
        """
        prior = {"prr": 8.0, "prr_lower": 5.5, "prr_upper": 11.5}
        # current CI [0.9, 2.1] — upper=2.1 < prior_lower=5.5 → DISMISSED
        assert self._classify(1.5, c_lo=0.9, c_up=2.1, prior=prior) == "DISMISSED"
