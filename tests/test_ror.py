"""
Tests for the ROR (Reporting Odds Ratio) estimator.

ROR = (a·d)/(b·c)  —  WHO/Uppsala standard alongside PRR.
CI: log-normal, SE = sqrt(1/a + 1/b + 1/c + 1/d).

For large n where the reaction is rare (c << c+d), ROR ≈ PRR asymptotically.
This test suite validates the formula, CI properties, and the ROR≈PRR convergence.
"""
import math
import pytest
from agent.tools.prr import _ror_ci, _prr_ci, _yates_chi2


class TestRORFormula:
    def test_null_association_gives_ror_one(self):
        """Equal proportions across groups → ROR = 1.0."""
        # a/(a+b) == c/(c+d) → PRR = 1, ROR = 1
        ror, lo, hi = _ror_ci(10, 90, 10, 90)
        assert ror == pytest.approx(1.0, rel=1e-4)

    def test_ror_above_prr_for_common_reactions(self):
        """
        ROR > PRR when the reaction is common in both arms.
        For rare reactions they converge; for common ones ROR inflates.
        """
        # drug: 40/100 = 40%, non-drug: 10/100 = 10%  → both arms high prevalence
        a, b, c, d = 40, 60, 10, 90
        ror, _, _ = _ror_ci(a, b, c, d)
        prr_lo, _ = _prr_ci(a, b, c, d)
        prr = (a / (a + b)) / (c / (c + d))
        assert ror > prr

    def test_ror_converges_to_prr_for_rare_reaction(self):
        """
        For rare reactions (small c relative to c+d), ROR ≈ PRR.
        This is the core asymptotic property — verifies they agree on FAERS data
        where most reactions are rare.
        """
        # drug: 100/10000 = 1%, non-drug: 50/990000 ≈ 0.005%  → very rare
        a, b, c, d = 100, 9900, 50, 989950
        ror, _, _ = _ror_ci(a, b, c, d)
        prr = (a / (a + b)) / (c / (c + d))
        # Within 2% of each other
        assert abs(ror - prr) / prr < 0.02

    def test_ci_contains_ror(self):
        """Lower CI < ROR < Upper CI always."""
        for (a, b, c, d) in [(10, 90, 5, 995), (100, 400, 50, 9950), (3, 17, 500, 9500)]:
            ror, lo, hi = _ror_ci(a, b, c, d)
            assert lo < ror < hi, f"CI does not contain ROR for ({a},{b},{c},{d})"

    def test_ci_narrows_with_larger_n(self):
        """More data → tighter CI on log scale."""
        _, lo_s, hi_s = _ror_ci(10, 40, 5, 995)
        _, lo_l, hi_l = _ror_ci(1000, 4000, 500, 99500)
        width_small = math.log(hi_s / lo_s)
        width_large = math.log(hi_l / lo_l)
        assert width_large < width_small

    def test_haldane_for_zero_cell(self):
        """Zero in any cell → Haldane +0.5, no crash."""
        ror, lo, hi = _ror_ci(0, 100, 50, 9950)
        assert ror >= 0.0
        assert lo < hi

    def test_ror_returns_three_values(self):
        """_ror_ci returns (ror, lower, upper) — not a pair."""
        result = _ror_ci(10, 90, 5, 995)
        assert len(result) == 3

    def test_signal_above_threshold(self):
        """A clear disproportionality signal should have ROR ≥ 2."""
        # drug: 50/100 = 50%, non-drug: 10/1000 = 1%  → strong signal
        ror, lo, hi = _ror_ci(50, 50, 10, 990)
        assert ror >= 2.0
        assert lo > 1.0


class TestPRRRORRelationship:
    """
    Analytical relationship between PRR and ROR:
    ROR = PRR · (1 − drug_rate) / (1 − bg_rate)
    When both rates are small: ROR ≈ PRR.
    """

    def test_algebraic_relationship(self):
        """
        ROR = (a·d)/(b·c) = PRR · (d/(c+d)) / (b/(a+b))
             = PRR · (1-bg_rate) / (1-drug_rate)
        """
        a, b, c, d = 40, 60, 10, 90
        ror_calc, _, _ = _ror_ci(a, b, c, d)
        prr = (a / (a + b)) / (c / (c + d))
        drug_rate = a / (a + b)
        bg_rate   = c / (c + d)
        expected_ror = prr * (1 - bg_rate) / (1 - drug_rate)
        assert ror_calc == pytest.approx(expected_ror, rel=1e-3)

    def test_both_estimators_agree_signal_direction(self):
        """If PRR > 1, ROR > 1 (same direction — always)."""
        cases = [
            (10, 90, 5, 995),   # modest signal
            (50, 50, 10, 990),  # strong signal
            (5,  95, 50, 950),  # below null
        ]
        for (a, b, c, d) in cases:
            prr = (a / (a + b)) / (c / (c + d))
            ror, _, _ = _ror_ci(a, b, c, d)
            if prr > 1:
                assert ror > 1, f"PRR>1 but ROR≤1 for ({a},{b},{c},{d})"
            elif prr < 1:
                assert ror < 1, f"PRR<1 but ROR≥1 for ({a},{b},{c},{d})"
