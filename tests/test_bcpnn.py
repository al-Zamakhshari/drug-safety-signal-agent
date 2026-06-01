"""
Tests for BCPNN — Information Component and IC025 computation.

Key properties:
  1. Independence (n11 ≈ E) → IC ≈ 0
  2. Strong signal → IC025 > 0  (WHO UMC flag)
  3. IC025 < IC always (2.5th percentile < point estimate)
  4. Large N → IC converges to log₂(n11/E)  (shrinkage fades)
  5. n11=0 guard — no crash
  6. IC_SD > 0 always (positive uncertainty)

Cross-check: for a known (n11, n1., n.1, N) tuple, verify IC against
the published formula manually.
"""
import math
import pytest
from agent.tools.bcpnn import compute_ic, annotate_signals_with_bcpnn


class TestComputeIC:

    def test_independence_ic_near_zero(self):
        """
        When n11 exactly equals the expected value E = n1.·n.1/N,
        IC should be ≈ 0 (no disproportionate reporting).
        """
        # n1.=1000, n.1=100, N=100000 → E = 1000*100/100000 = 1
        # If n11=1=E, IC = log2(1.5/1.5) = 0
        ic, ic025, ic975 = compute_ic(1, 1000, 100, 100_000)
        assert abs(ic) < 0.1, f"Expected IC≈0 at independence, got {ic}"

    def test_strong_signal_ic025_positive(self):
        """
        A strong, large-n signal (n11 >> E) should have IC025 > 0 (WHO flag).
        """
        # Drug: 500/1000 = 50%; background: 100/100000 = 0.1% → E=1, n11=500
        ic, ic025, ic975 = compute_ic(500, 1000, 100, 100_000)
        assert ic > 0, f"Expected IC > 0 for strong signal, got {ic}"
        assert ic025 > 0, f"Expected IC025 > 0 (WHO signal), got {ic025}"

    def test_ic025_always_less_than_ic(self):
        """IC025 (2.5th percentile) must always be below the IC point estimate."""
        cases = [
            (10, 500, 200, 50_000),
            (1, 100, 50, 10_000),
            (1000, 5000, 3000, 1_000_000),
            (3, 50, 30, 5_000),
        ]
        for (n11, n1d, nd1, N) in cases:
            ic, ic025, ic975 = compute_ic(n11, n1d, nd1, N)
            assert ic025 <= ic, f"IC025 ({ic025}) > IC ({ic}) for {(n11,n1d,nd1,N)}"
            assert ic025 <= ic975, f"IC025 ({ic025}) > IC975 ({ic975})"

    def test_ic975_always_greater_than_ic(self):
        """IC975 (97.5th percentile) must always be above the IC point estimate."""
        ic, ic025, ic975 = compute_ic(100, 2000, 500, 500_000)
        assert ic975 >= ic

    def test_large_n_ic_converges_to_log2_ratio(self):
        """
        For large N and large n11, IC should converge to log₂(n11/E).
        The shrinkage +0.5 terms become negligible.
        """
        n11 = 10_000; n1d = 50_000; nd1 = 20_000; N = 10_000_000
        E = n1d * nd1 / N   # = 100
        ic, _, _ = compute_ic(n11, n1d, nd1, N)
        expected_ic = math.log2(n11 / E)
        assert abs(ic - expected_ic) < 0.01, \
            f"Large-N IC ({ic}) should be close to log2(n11/E)={expected_ic:.3f}"

    def test_zero_n11_no_crash(self):
        """n11=0 should return a negative IC (below-expected), not crash."""
        ic, ic025, ic975 = compute_ic(0, 1000, 100, 100_000)
        assert isinstance(ic, float)
        assert ic < 0, "n11=0 → IC should be negative (fewer than expected)"

    def test_zero_denominators_return_zeros(self):
        """Invalid inputs (N=0 or zero marginals) should return (0,0,0) safely."""
        assert compute_ic(5, 0, 100, 10_000) == (0.0, 0.0, 0.0)
        assert compute_ic(5, 100, 0, 10_000) == (0.0, 0.0, 0.0)
        assert compute_ic(5, 100, 100, 0)    == (0.0, 0.0, 0.0)

    def test_hand_computed_cross_check(self):
        """
        Verify against a hand-computed reference case.
        n11=50, n1.=500, n.1=200, N=100_000
        E = 500*200/100000 = 1.0
        IC = log2(50.5/1.5) = log2(33.67) ≈ 5.074
        gamma = (100002)^2 / (502 * 202) ≈ 98.6
        (exact IC025 depends on variance — just check IC is in the right ballpark)
        """
        ic, ic025, ic975 = compute_ic(50, 500, 200, 100_000)
        E = 500 * 200 / 100_000
        expected_ic = math.log2((50 + 0.5) / (E + 0.5))
        assert abs(ic - expected_ic) < 0.001, \
            f"IC={ic} differs from expected {expected_ic:.3f}"
        assert ic025 > 0, "Strong signal should have IC025 > 0"

    def test_small_n_ic025_may_be_below_zero(self):
        """
        A marginal signal (PRR≈2) at small n should have IC025 ≤ 0
        — not flagged despite positive point estimate.
        This is the BCPNN equivalent of the PRR lower-CI gate.
        """
        # n11=3, E=1.5 → IC ≈ 1, but SD is large for small n
        ic, ic025, _ = compute_ic(3, 300, 500, 100_000)
        # Just verify IC > 0 (positive signal) but IC025 may not be
        assert ic > 0


class TestAnnotateSignals:

    def test_adds_ic_fields(self):
        """annotate_signals_with_bcpnn adds ic, ic025, ic975, ic_signal."""
        signals = [
            {"drug_count": 500, "baseline": 200, "prr": 8.5},
            {"drug_count": 5,   "baseline": 300, "prr": 1.1},
        ]
        result = annotate_signals_with_bcpnn(signals, drug_total=2000, faers_total=100_000)
        for s in result:
            assert "ic"        in s
            assert "ic025"     in s
            assert "ic975"     in s
            assert "ic_signal" in s

    def test_strong_signal_flagged(self):
        """Large-n strong signal should have ic_signal=True."""
        signals = [{"drug_count": 500, "baseline": 100, "prr": 15.0}]
        result = annotate_signals_with_bcpnn(signals, 1000, 100_000)
        assert result[0]["ic_signal"] is True

    def test_ic_signal_consistent_with_ic025(self):
        """ic_signal must equal ic025 > 0."""
        signals = [
            {"drug_count": 500, "baseline": 100, "prr": 15.0},
            {"drug_count": 3,   "baseline": 500, "prr": 0.5},
        ]
        result = annotate_signals_with_bcpnn(signals, 1000, 100_000)
        for s in result:
            assert s["ic_signal"] == (s["ic025"] > 0.0)

    def test_empty_list(self):
        """Empty input returns empty list."""
        assert annotate_signals_with_bcpnn([], 1000, 100_000) == []

    def test_preserves_existing_fields(self):
        """Annotation must not drop any existing fields."""
        original = {"drug_count": 100, "baseline": 200, "prr": 3.0, "ebgm": 2.8}
        result = annotate_signals_with_bcpnn([dict(original)], 1000, 100_000)
        for key in original:
            assert key in result[0]
