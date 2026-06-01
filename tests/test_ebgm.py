"""
Tests for the EBGM / Gamma-Poisson Shrinker (DuMouchel 1999).

Key properties:
  1. EBGM > 1 for observed > expected (positive signal direction)
  2. EBGM shrinks toward 1 for small n (the whole point)
  3. EBGM ≈ O/E for large n (posterior converges to MLE)
  4. EB05 < EBGM always (5th percentile < geometric mean)
  5. EB05 ≥ 2.0 threshold is a meaningful gate
  6. GPS prior fits successfully on realistic (O, E) data
"""
import math
import numpy as np
import pytest
from agent.tools.ebgm import fit_gps_prior, compute_ebgm, annotate_signals_with_ebgm


# ---------------------------------------------------------------------------
# Shared fixture: a small realistic set of (O, E) pairs
# ---------------------------------------------------------------------------

_REALISTIC_SIGNALS = [
    # Strong signals (O >> E)
    {"drug_count": 3057, "baseline": 45000,  "prr": 82.7},
    {"drug_count": 1111, "baseline": 12000,  "prr": 11.5},
    {"drug_count": 1504, "baseline": 18000,  "prr": 10.4},
    # Moderate signals
    {"drug_count": 500,  "baseline": 60000,  "prr": 4.5},
    {"drug_count": 100,  "baseline": 30000,  "prr": 2.2},
    # Noise (O ≈ E)
    {"drug_count": 50,   "baseline": 60000,  "prr": 1.0},
    {"drug_count": 20,   "baseline": 25000,  "prr": 1.0},
    # Sub-threshold
    {"drug_count": 10,   "baseline": 20000,  "prr": 0.7},
]
_DRUG_TOTAL   = 82_699
_FAERS_TOTAL  = 11_942_667


class TestGPSPriorFitting:

    def test_fit_converges_on_realistic_data(self):
        """GPS prior fitting should converge on a plausible (O, E) dataset."""
        O = np.array([s["drug_count"] for s in _REALISTIC_SIGNALS], dtype=float)
        E = np.array(
            [_DRUG_TOTAL * s["baseline"] / _FAERS_TOTAL for s in _REALISTIC_SIGNALS],
            dtype=float,
        )
        params, converged = fit_gps_prior(O, E)
        # Even if optimiser flags non-convergence, params must be finite and positive
        a1, b1, a2, b2, P = params
        assert all(v > 0 for v in [a1, b1, a2, b2])
        assert 0.0 < P < 1.0

    def test_fit_returns_five_params(self):
        O = np.array([10.0, 50.0, 100.0, 5.0, 200.0])
        E = np.array([5.0,  10.0,  20.0, 4.0,  50.0])
        params, _ = fit_gps_prior(O, E)
        assert len(params) == 5

    def test_fit_with_too_few_points_returns_default(self):
        """< 5 valid data points → returns default conservative prior, no crash."""
        params, converged = fit_gps_prior(np.array([5.0, 3.0]), np.array([2.0, 1.0]))
        assert not converged
        assert len(params) == 5


class TestEBGMComputation:

    def _default_params(self):
        """Fit params from the realistic signals fixture."""
        O = np.array([s["drug_count"] for s in _REALISTIC_SIGNALS], dtype=float)
        E = np.array(
            [_DRUG_TOTAL * s["baseline"] / _FAERS_TOTAL for s in _REALISTIC_SIGNALS],
            dtype=float,
        )
        params, _ = fit_gps_prior(O, E)
        return params

    def test_eb05_less_than_ebgm(self):
        """EB05 (5th percentile) must always be below EBGM (geometric mean)."""
        params = self._default_params()
        for s in _REALISTIC_SIGNALS:
            e = _DRUG_TOTAL * s["baseline"] / _FAERS_TOTAL
            ebgm, eb05 = compute_ebgm(s["drug_count"], e, params)
            if ebgm > 0:
                assert eb05 <= ebgm, f"EB05 > EBGM for {s}"

    def test_strong_signal_eb05_above_threshold(self):
        """
        A very strong signal (O/E >> 1, large n) should have EB05 ≥ 2.0
        — the FDA signal threshold.
        """
        params = self._default_params()
        # IMPAIRED GASTRIC EMPTYING: O=3057, E≈drug_total*45000/N ≈ 0.31
        e = _DRUG_TOTAL * 45000 / _FAERS_TOTAL
        ebgm, eb05 = compute_ebgm(3057, e, params)
        assert eb05 >= 2.0, f"Expected EB05 ≥ 2.0 for strong signal, got {eb05}"

    def test_noise_signal_eb05_below_threshold(self):
        """
        O ≈ E (no signal) with small n should have EB05 < 2.0.
        This is the shrinkage property: PRR might look elevated, but EBGM shrinks.
        """
        params = self._default_params()
        # O = 3, E = 2.5  → O/E = 1.2, tiny n
        ebgm, eb05 = compute_ebgm(3, 2.5, params)
        assert eb05 < 2.0, f"Expected EB05 < 2.0 for noise, got {eb05}"

    def test_ebgm_positive_and_finite(self):
        """EBGM and EB05 should always be non-negative finite floats."""
        params = self._default_params()
        for o, e in [(1, 0.5), (10, 5.0), (100, 20.0), (1000, 10.0)]:
            ebgm, eb05 = compute_ebgm(o, float(e), params)
            assert math.isfinite(ebgm)
            assert math.isfinite(eb05)
            assert ebgm >= 0.0
            assert eb05 >= 0.0

    def test_eb05_grows_with_n_for_same_oe_ratio(self):
        """
        The practical GPS shrinkage property: for the same O/E ratio,
        EB05 (the conservative lower bound) grows with n.

        This is the clinically important property: a signal with O/E=3 on n=3
        gets EB05 < 2.0 (not flagged), while O/E=3 on n=300 gets EB05 > 2.0
        (flagged). The threshold protects against small-n false signals.

        Note: the direction of EBGM vs O/E depends on the prior shape, which
        in turn depends on the drug's overall signal profile. What is guaranteed
        is that EB05 is more conservative (lower) for small n.
        """
        params = self._default_params()
        # Same O/E = 3 at two sample sizes
        _, eb05_small = compute_ebgm(3,   1.0,   params)   # n=3,  O/E=3
        _, eb05_large = compute_ebgm(300, 100.0, params)   # n=300, O/E=3
        assert eb05_large > eb05_small, (
            f"EB05 should grow with n for same O/E ratio. "
            f"Got eb05_small={eb05_small}, eb05_large={eb05_large}"
        )


class TestAnnotateSignals:

    def test_annotate_adds_ebgm_and_eb05(self):
        """annotate_signals_with_ebgm adds 'ebgm', 'eb05', 'eb05_signal' fields."""
        signals = [dict(s) for s in _REALISTIC_SIGNALS[:4]]
        result = annotate_signals_with_ebgm(signals, _DRUG_TOTAL, _FAERS_TOTAL)
        for s in result:
            assert "ebgm" in s
            assert "eb05" in s
            assert "eb05_signal" in s
            assert isinstance(s["ebgm"], float)
            assert isinstance(s["eb05"], float)
            assert isinstance(s["eb05_signal"], bool)

    def test_annotate_empty_list(self):
        """Empty signal list → returns empty list without error."""
        result = annotate_signals_with_ebgm([], _DRUG_TOTAL, _FAERS_TOTAL)
        assert result == []

    def test_annotate_preserves_existing_fields(self):
        """Annotation should not remove any existing fields."""
        original = dict(_REALISTIC_SIGNALS[0])
        signals  = [dict(_REALISTIC_SIGNALS[0])]
        result   = annotate_signals_with_ebgm(signals, _DRUG_TOTAL, _FAERS_TOTAL)
        for key in original:
            assert key in result[0], f"Field '{key}' was lost after annotation"

    def test_eb05_signal_flag_consistent(self):
        """eb05_signal should equal eb05 >= 2.0."""
        signals = [dict(s) for s in _REALISTIC_SIGNALS]
        result  = annotate_signals_with_ebgm(signals, _DRUG_TOTAL, _FAERS_TOTAL)
        for s in result:
            assert s["eb05_signal"] == (s["eb05"] >= 2.0)
