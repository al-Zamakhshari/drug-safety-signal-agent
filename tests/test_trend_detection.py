"""
Tests for data-relative trend detection in anomaly_signals.

The key regression: before the fix, trend windows were hardcoded to
2020-01-01 / 2023-01-01 cutoffs. For rofecoxib (withdrawn 2004) and any
other pre-2020 drug, both windows were empty → trend was always STABLE
regardless of the actual time pattern. The fix uses first-third / last-third
of the drug's own observed quarter range.
"""
import pytest


def _apply_trend_logic(strata: list[dict]) -> str:
    """
    Replicate the trend logic from get_anomaly_signals exactly,
    so we can unit-test it without OpenSearch.
    """
    from agent.tools.anomaly_signals import _mh_rate_ratio

    quarters_sorted = sorted(s["quarter"] for s in strata)
    n_q = len(quarters_sorted)
    if n_q >= 3:
        early_cutoff  = quarters_sorted[n_q // 3]
        recent_cutoff = quarters_sorted[2 * n_q // 3]
        early_strata  = [s for s in strata if s["quarter"] <  early_cutoff]
        recent_strata = [s for s in strata if s["quarter"] >= recent_cutoff]
    elif n_q == 2:
        early_strata  = [strata[0]]
        recent_strata = [strata[1]]
    else:
        early_strata  = []
        recent_strata = strata

    r_rr, r_lo, _ = _mh_rate_ratio(recent_strata) if recent_strata else (0, 0, 0)
    e_rr, e_lo, _ = _mh_rate_ratio(early_strata)  if early_strata  else (0, 0, 0)

    if e_lo <= 1.0 and r_lo > 1.0:
        return "EMERGING"
    elif r_lo > 1.0 and r_rr > e_rr * 1.5 and e_rr > 0:
        return "GROWING"
    else:
        return "STABLE"


def _stratum(quarter: str, a: int, n1: int, c: int, n2: int) -> dict:
    return {"quarter": quarter, "drug_count": a, "drug_total": n1,
            "comp_count": c, "comp_total": n2}


class TestDataRelativeTrendWindows:

    def test_growing_signal_recent_drug(self):
        """Signal present early and accelerating recently → GROWING."""
        strata = [
            # Early quarters — modest signal
            _stratum("2018-01-01", 5,  500, 10, 5000),
            _stratum("2018-04-01", 6,  500, 10, 5000),
            _stratum("2018-07-01", 7,  500, 11, 5000),
            _stratum("2018-10-01", 8,  500, 11, 5000),
            # Middle quarters
            _stratum("2019-01-01", 10, 500, 12, 5000),
            _stratum("2019-04-01", 12, 500, 12, 5000),
            # Recent quarters — strong acceleration
            _stratum("2019-07-01", 40, 500, 10, 5000),
            _stratum("2019-10-01", 50, 500, 10, 5000),
            _stratum("2020-01-01", 60, 500, 10, 5000),
        ]
        trend = _apply_trend_logic(strata)
        assert trend == "GROWING", f"Expected GROWING, got {trend}"

    def test_emerging_signal_absent_then_present(self):
        """No signal early, robust signal recently → EMERGING."""
        strata = [
            # Early quarters — no signal (c >> a)
            _stratum("2018-01-01", 1, 500, 80, 5000),
            _stratum("2018-04-01", 1, 500, 80, 5000),
            _stratum("2018-07-01", 2, 500, 80, 5000),
            # Recent quarters — clear signal
            _stratum("2019-07-01", 50, 500, 10, 5000),
            _stratum("2019-10-01", 55, 500, 10, 5000),
            _stratum("2020-01-01", 60, 500, 10, 5000),
        ]
        trend = _apply_trend_logic(strata)
        assert trend == "EMERGING", f"Expected EMERGING, got {trend}"

    def test_pre_2020_drug_not_always_stable(self):
        """
        Regression: a pre-2020 drug (e.g. rofecoxib 2001-2004) with a
        genuinely growing signal must NOT always return STABLE.
        The old hardcoded 2020/2023 cutoffs made this impossible.
        """
        strata = [
            # Early quarters (2001) — modest signal
            _stratum("2001-01-01", 5,  200, 15, 3000),
            _stratum("2001-04-01", 6,  200, 15, 3000),
            _stratum("2001-07-01", 7,  200, 14, 3000),
            # Middle
            _stratum("2002-01-01", 10, 200, 14, 3000),
            _stratum("2002-04-01", 12, 200, 13, 3000),
            # Recent (2002-2003) — strong acceleration
            _stratum("2003-01-01", 45, 200, 10, 3000),
            _stratum("2003-04-01", 50, 200, 10, 3000),
            _stratum("2003-07-01", 60, 200, 9,  3000),
            _stratum("2004-01-01", 70, 200, 9,  3000),
        ]
        trend = _apply_trend_logic(strata)
        # With the old hardcoded windows (2020/2023), all strata < 2020 → STABLE always
        # With data-relative windows, this correctly detects GROWING
        assert trend in ("GROWING", "EMERGING"), \
            f"Pre-2020 drug with accelerating signal should not be STABLE, got {trend}"

    def test_stable_signal_stays_stable(self):
        """Consistent signal across all quarters → STABLE."""
        strata = [
            _stratum(f"200{y}-{m:02d}-01", 20, 500, 10, 5000)
            for y in range(1, 5)
            for m in [1, 4, 7, 10]
        ]
        trend = _apply_trend_logic(strata)
        assert trend == "STABLE"

    def test_single_stratum_no_crash(self):
        """Only one quarterly stratum should not crash."""
        strata = [_stratum("2022-01-01", 20, 500, 10, 5000)]
        trend = _apply_trend_logic(strata)
        assert trend in ("STABLE", "EMERGING", "GROWING")

    def test_two_strata_uses_both(self):
        """Two strata: first is early, second is recent — check directionality."""
        strata = [
            _stratum("2020-01-01", 2,  500, 80, 5000),   # weak early
            _stratum("2023-01-01", 50, 500, 10, 5000),   # strong recent
        ]
        trend = _apply_trend_logic(strata)
        assert trend == "EMERGING"
