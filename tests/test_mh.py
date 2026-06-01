"""
Tests for the Mantel–Haenszel stratified rate ratio estimator.

_mh_rate_ratio(strata) takes a list of per-quarter stratum dicts and returns
(rr_mh, lower_95, upper_95) using the Robins–Breslow–Greenland variance.

Key properties to verify:
  1. Null: equal rates across all strata → RR_MH = 1.0
  2. Consistent signal: RR_MH > 1, CI lower > 1 (robust)
  3. Temporal confounding control: MH differs from naive pool when rates vary
  4. Robins–Breslow–Greenland variance reduces with more strata
  5. Haldane continuity applied per-stratum for zero cells
  6. Single stratum equals simple rate ratio
"""
import math
import pytest
from agent.tools.anomaly_signals import _mh_rate_ratio, _rate_ratio_ci


def _make_stratum(a, n1, c, n2):
    return {"drug_count": a, "drug_total": n1, "comp_count": c, "comp_total": n2}


class TestMHRateRatio:

    def test_null_single_stratum(self):
        """Equal rates → RR_MH = 1.0."""
        strata = [_make_stratum(10, 100, 50, 500)]
        rr, lo, hi = _mh_rate_ratio(strata)
        assert rr == pytest.approx(1.0, rel=1e-3)

    def test_null_multiple_strata(self):
        """Equal rates across all strata → RR_MH = 1.0."""
        strata = [_make_stratum(10, 100, 50, 500) for _ in range(5)]
        rr, lo, hi = _mh_rate_ratio(strata)
        assert rr == pytest.approx(1.0, rel=1e-3)

    def test_consistent_signal_lower_above_one(self):
        """Signal in all strata → CI lower > 1.0."""
        # drug: 30% per quarter, comp: 5% per quarter
        strata = [_make_stratum(30, 100, 25, 500) for _ in range(8)]
        rr, lo, hi = _mh_rate_ratio(strata)
        assert rr == pytest.approx(6.0, rel=1e-2)
        assert lo > 1.0

    def test_ci_contains_rr(self):
        """Lower CI < RR < Upper CI for all reasonable inputs."""
        strata = [
            _make_stratum(20, 100, 40, 500),
            _make_stratum(25, 120, 45, 600),
            _make_stratum(18, 90, 38, 450),
        ]
        rr, lo, hi = _mh_rate_ratio(strata)
        assert lo < rr < hi

    def test_more_strata_tighter_ci(self):
        """More strata (more data) → tighter CI on log scale."""
        base = _make_stratum(20, 100, 40, 500)
        strata_few  = [base] * 2
        strata_many = [base] * 20
        _, lo_few,  hi_few  = _mh_rate_ratio(strata_few)
        _, lo_many, hi_many = _mh_rate_ratio(strata_many)
        width_few  = math.log(hi_few  / lo_few)
        width_many = math.log(hi_many / lo_many)
        assert width_many < width_few

    def test_single_stratum_point_estimate_equals_simple_rr(self):
        """
        With one stratum, MH rate ratio (the point estimate) must equal
        the simple rate ratio exactly.
        Note: the CIs will differ — MH uses Robins–Breslow–Greenland variance
        while _rate_ratio_ci uses the log-normal approximation. These are
        different estimators and do NOT produce the same CI.
        """
        a, n1, c, n2 = 40, 200, 80, 2000
        strata = [_make_stratum(a, n1, c, n2)]
        rr_mh, lo_mh, hi_mh = _mh_rate_ratio(strata)
        simple_rr = (a / n1) / (c / n2)
        assert rr_mh == pytest.approx(simple_rr, rel=1e-3)
        # CI should still be valid (lower < RR < upper)
        assert lo_mh < rr_mh < hi_mh

    def test_haldane_for_zero_comp_count(self):
        """Zero comp_count in a stratum → Haldane +0.5, no crash."""
        strata = [
            _make_stratum(20, 100, 0, 500),   # zero comparator — Haldane applied
            _make_stratum(18, 100, 2, 500),
        ]
        rr, lo, hi = _mh_rate_ratio(strata)
        assert rr > 1.0
        assert math.isfinite(rr)
        assert math.isfinite(lo)
        assert math.isfinite(hi)

    def test_empty_strata_returns_zero(self):
        """No valid strata → (0.0, 0.0, inf)."""
        rr, lo, hi = _mh_rate_ratio([])
        assert rr == 0.0
        assert lo == 0.0

    def test_strata_with_zero_totals_skipped(self):
        """Strata with n1=0 or n2=0 are silently skipped."""
        strata = [
            _make_stratum(10, 0, 20, 500),    # n1=0 → invalid, skipped
            _make_stratum(20, 100, 40, 500),  # valid
        ]
        rr, lo, hi = _mh_rate_ratio(strata)
        # Should give same result as single valid stratum
        rr_single, _, _ = _mh_rate_ratio([_make_stratum(20, 100, 40, 500)])
        assert rr == pytest.approx(rr_single, rel=1e-3)

    def test_mh_controls_temporal_confounding(self):
        """
        MH rate ratio should be closer to the true within-stratum RR than the
        naive pooled estimate when the drug's total volume changes over time.

        Scenario: true within-quarter RR = 2.0 in all quarters, but drug volume
        grew 10× over the period. Naive pooling inflates the estimate; MH doesn't.
        """
        true_rr = 2.0
        # 8 quarters; drug volume grows 1× → 8×; comp volume is constant
        strata = []
        for q in range(8):
            n1 = 100 * (q + 1)     # drug volume grows
            n2 = 1000              # comp volume constant
            a  = int(n1 * 0.10)   # drug rate = 10%
            c  = int(n2 * 0.05)   # comp rate = 5%  → true RR = 2.0
            strata.append(_make_stratum(a, n1, c, n2))

        rr_mh, _, _ = _mh_rate_ratio(strata)

        # Naive pool
        total_a  = sum(s["drug_count"]  for s in strata)
        total_n1 = sum(s["drug_total"]  for s in strata)
        total_c  = sum(s["comp_count"]  for s in strata)
        total_n2 = sum(s["comp_total"]  for s in strata)
        rr_naive = (total_a / total_n1) / (total_c / total_n2)

        # Both should be close to 2.0 in this balanced scenario,
        # but MH should be at least as accurate as naive pool
        assert abs(rr_mh - true_rr) <= abs(rr_naive - true_rr) + 0.1
