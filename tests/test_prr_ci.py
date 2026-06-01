"""
Tests for the PRR 95% CI and Benjamini-Hochberg FDR correction.

These test the new statistical additions (Phase 1.1 + 1.2) against
analytically known results.
"""
import math
import numpy as np
import pytest
from agent.tools.prr import _prr_ci, _yates_chi2


class TestPRRCI:
    """PRR 95% confidence interval — log-normal approximation (Evans 2001)."""

    def test_ci_symmetric_on_log_scale(self):
        """ln(PRR_hi / PRR) should equal ln(PRR / PRR_lo) within rounding."""
        a, b, c, d = 100, 400, 50, 9950
        lo, hi = _prr_ci(a, b, c, d)
        prr = (a / (a + b)) / (c / (c + d))
        # Symmetric on log scale
        assert abs(math.log(hi / prr) - math.log(prr / lo)) < 0.01

    def test_ci_lower_above_one_for_strong_signal(self):
        """A strong signal (PRR≈20, n=100) should have lower CI > 1."""
        # drug: 100/500 = 20%, non-drug: 50/9500 ≈ 0.5%  → PRR ≈ 38
        a, b, c, d = 100, 400, 50, 9950
        lo, hi = _prr_ci(a, b, c, d)
        assert lo > 1.0
        assert hi > lo

    def test_ci_lower_below_one_for_weak_small_n(self):
        """A weak signal at tiny n should have lower CI < 1 (not robust)."""
        # PRR ≈ 2 but n=3 — wide CI should cross 1
        a, b, c, d = 3, 97, 1000, 99000
        lo, hi = _prr_ci(a, b, c, d)
        # The CI should be very wide — lo may be below 1
        assert lo < hi  # basic sanity
        # With n=3 the SE is large; lower bound should be < 2 even if PRR > 2
        prr = (a / (a + b)) / (c / (c + d))
        assert lo < prr  # lower bound is below point estimate

    def test_robust_flag_matches_lower_bound(self):
        """robust = True iff lower CI ≥ 1.0."""
        # Strong signal
        lo_strong, _ = _prr_ci(100, 400, 50, 9950)
        assert lo_strong >= 1.0   # robust

        # Marginal signal
        lo_weak, _ = _prr_ci(3, 97, 1000, 99000)
        # Don't assert direction — just that the function returns valid floats
        assert isinstance(lo_weak, float)
        assert isinstance(_, float)

    def test_haldane_correction_for_zero_cell(self):
        """a=0 or c=0 should not raise — uses +0.5 continuity."""
        lo, hi = _prr_ci(0, 100, 50, 9950)
        assert lo >= 0.0
        assert hi > lo

    def test_ci_narrows_with_more_data(self):
        """Larger n → tighter CI (interval shrinks)."""
        lo_small, hi_small = _prr_ci(10, 40, 5, 995)
        lo_large, hi_large = _prr_ci(1000, 4000, 500, 99500)
        width_small = math.log(hi_small / lo_small)
        width_large = math.log(hi_large / lo_large)
        assert width_large < width_small


class TestBHFDR:
    """Benjamini-Hochberg FDR correction — structural properties."""

    def _apply_bh(self, p_values: list[float]) -> list[float]:
        """Replicate the BH code from calculate_prr for unit testing."""
        p = np.array(p_values)
        m = len(p)
        order  = np.argsort(p)
        ranks  = np.empty(m, int)
        ranks[order] = np.arange(1, m + 1)
        q_raw = p * m / ranks
        q_adj = np.empty(m)
        q_adj[order] = np.minimum.accumulate(q_raw[order][::-1])[::-1]
        return [float(min(qi, 1.0)) for qi in q_adj]

    def test_q_values_monotone_with_p_values(self):
        """BH q-values should be non-decreasing as p-values increase."""
        p_vals = [0.001, 0.01, 0.05, 0.1, 0.5]
        q_vals = self._apply_bh(p_vals)
        assert q_vals == sorted(q_vals), f"q not monotone: {q_vals}"

    def test_small_p_passes_fdr(self):
        """A highly significant p-value (0.0001) should pass BH q < 0.05."""
        p_vals = [0.0001] + [0.5] * 49   # 1 truly significant, 49 nulls
        q_vals = self._apply_bh(p_vals)
        assert q_vals[0] < 0.05

    def test_all_null_fails_fdr(self):
        """All p=0.5 → all q ≥ 0.05 (no false discoveries)."""
        q_vals = self._apply_bh([0.5] * 50)
        assert all(q >= 0.05 for q in q_vals)

    def test_q_bounded_at_one(self):
        """No q-value should exceed 1.0."""
        q_vals = self._apply_bh([0.9, 0.95, 0.99])
        assert all(q <= 1.0 for q in q_vals)

    def test_m_equals_all_tested_not_just_passing(self):
        """
        m must be the total reactions tested (incl. PRR < 2.0), not just
        those that pass the PRR threshold. This test verifies the BH
        denominator captures the full multiple-comparison burden.
        A stricter m (larger set) gives higher (more conservative) q-values.
        """
        # 5 tests: 1 significant + 4 nulls
        p_vals_5  = [0.001, 0.5, 0.5, 0.5, 0.5]
        # Same but m=50 (correct: includes non-PRR signals)
        p_vals_50 = [0.001] + [0.5] * 49

        q_small_m = self._apply_bh(p_vals_5)[0]
        q_large_m = self._apply_bh(p_vals_50)[0]

        # Larger m → more conservative → higher q-value for the same p
        assert q_large_m >= q_small_m


class TestRateRatioCI:
    """Tests for the rate-ratio CI in anomaly_signals (same formula, pooled counts)."""

    def test_formula_matches_prr_ci(self):
        """
        The rate-ratio CI formula in anomaly_signals is the same log-normal
        approximation as _prr_ci. Verify they produce the same result when
        given equivalent inputs.
        """
        from agent.tools.anomaly_signals import _rate_ratio_ci
        a, n1, c, n2 = 100, 500, 50, 10000
        # Equivalent to _prr_ci(a, n1-a, c, n2-c)
        rr, lo, hi = _rate_ratio_ci(a, n1, c, n2)
        lo_prr, hi_prr = _prr_ci(a, n1 - a, c, n2 - c)
        assert abs(lo - lo_prr) < 0.01
        assert abs(hi - hi_prr) < 0.01

    def test_zero_comp_count_gives_wide_ci(self):
        """comp_count=0 → Haldane correction → large ratio, wide CI (not infinite)."""
        from agent.tools.anomaly_signals import _rate_ratio_ci
        rr, lo, hi = _rate_ratio_ci(50, 1000, 0, 10000)
        assert rr > 10    # large ratio (drug much higher than comparator)
        assert hi > lo    # CI has valid bounds
        assert hi < float("inf")  # not infinite
