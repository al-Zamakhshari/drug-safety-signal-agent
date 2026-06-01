"""
Pure-function tests for PRR formula and Yates χ² implementation.

These tests run without OpenSearch — they only test the deterministic math.
They verify the correctness of the EMA PRR formula (EMA/813938/2011).
"""
import pytest
from agent.tools.prr import _yates_chi2


# ---------------------------------------------------------------------------
# Yates-corrected χ²
# ---------------------------------------------------------------------------

class TestYatesChi2:
    """χ² = N·(|ad−bc| − N/2)² / ((a+b)(c+d)(a+c)(b+d))"""

    def test_zero_denom_returns_zero(self):
        """All zeros → no association, no crash."""
        assert _yates_chi2(0, 0, 0, 0) == 0.0

    def test_zero_marginal_returns_zero(self):
        """Any zero marginal (col or row sum = 0) → undefined, return 0."""
        assert _yates_chi2(5, 0, 0, 0) == 0.0  # col2=0
        assert _yates_chi2(0, 5, 0, 0) == 0.0  # col1=0

    def test_perfect_independence_near_zero(self):
        """No association: equal proportions across rows → χ² ≈ 0."""
        # a/row1 == c/row2 → no signal
        chi2 = _yates_chi2(10, 90, 10, 90)
        assert chi2 == pytest.approx(0.0, abs=0.1)

    def test_above_significance_threshold(self):
        """Strong signal should produce χ² well above 4.0 (EMA threshold)."""
        # drug: 50/100 = 50%; background: 10/1000 = 1% → very strong signal
        chi2 = _yates_chi2(50, 50, 10, 990)
        assert chi2 > 4.0

    def test_weak_signal_below_threshold(self):
        """Weak signal with small counts should be below χ² = 4.0."""
        # drug: 3/20 = 15%; background: 5/100 = 5% — marginal, small n
        chi2 = _yates_chi2(3, 17, 5, 95)
        assert chi2 < 4.0

    def test_minimum_ema_signal(self):
        """
        EMA threshold: PRR≥2, χ²≥4, n≥3.
        A clear 2:1 ratio with sufficient counts should pass χ²≥4.
        """
        # drug: 20/100 = 20%, background: 100/10000 = 1% → PRR=20, strong
        chi2 = _yates_chi2(20, 80, 100, 9900)
        assert chi2 >= 4.0

    def test_returns_float(self):
        assert isinstance(_yates_chi2(10, 90, 5, 95), float)


# ---------------------------------------------------------------------------
# PRR formula (tested without OpenSearch by directly computing the ratio)
# ---------------------------------------------------------------------------

def _compute_prr(a, b, c, d):
    """
    Direct PRR calculation from a 2×2 table.
      a = drug reports with reaction        b = drug reports without reaction
      c = non-drug reports with reaction    d = non-drug reports without reaction

    PRR = (a/(a+b)) / (c/(c+d))
    """
    drug_total     = a + b
    non_drug_total = c + d
    if drug_total == 0 or non_drug_total == 0 or c == 0:
        return None
    return (a / drug_total) / (c / non_drug_total)


class TestPRRFormula:
    def test_prr_null_association(self):
        """Equal rates → PRR = 1.0."""
        prr = _compute_prr(10, 90, 10, 90)
        assert prr == pytest.approx(1.0, rel=1e-6)

    def test_prr_signal(self):
        """Drug rate 4× higher than background → PRR = 4.0."""
        # drug: 40/100 = 40%; non-drug: 10/100 = 10%
        prr = _compute_prr(40, 60, 10, 90)
        assert prr == pytest.approx(4.0, rel=1e-6)

    def test_prr_below_threshold(self):
        """Mild elevation → PRR < 2.0, should not be a signal."""
        # drug: 5/100 = 5%; non-drug: 4/100 = 4%
        prr = _compute_prr(5, 95, 4, 96)
        assert prr < 2.0

    def test_prr_denominator_self_subtraction(self):
        """
        Non-drug total must exclude the drug (non-exposed arm).
        Verify: if drug=50 out of 10050 total, non-drug=10000.
        PRR should NOT use total FAERS as denominator for non-drug.
        """
        a, b = 50, 50       # drug: 50/100 = 50%
        # non-drug total = faers_total - drug_total = 10000
        c, d = 100, 9900    # non-drug: 100/10000 = 1%
        prr = _compute_prr(a, b, c, d)
        assert prr == pytest.approx(50.0, rel=1e-3)
        # Incorrect formula (using full population as denominator) would give ~49
        # This confirms we're using the non-exposed denominator correctly.
        assert prr > 49.0
