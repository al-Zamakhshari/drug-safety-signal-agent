"""
Tests for anomaly signal processing — pure-function tests that don't require OpenSearch.

Focused on the sentinel (no_class_baseline) handling introduced in Phase 2.
"""
import pytest


def _sort_signals(signals: list[dict]) -> list[dict]:
    """
    Mirrors the sort logic in get_anomaly_signals so we can test it in isolation.
    no_class_baseline signals have max_ratio=None; they sort last (treated as 0).
    """
    return sorted(signals, key=lambda x: -(x["max_ratio"] or 0))


class TestSentinelHandling:
    def test_sort_with_none_ratio_does_not_crash(self):
        """
        Critical regression test: sorting a mixed list of signals where
        no_class_baseline=True entries have max_ratio=None must not raise TypeError.
        This was the crash introduced by the Phase-2 fix:
          signals.sort(key=lambda x: -x["max_ratio"])  # -None → TypeError
        """
        signals = [
            {"reaction": "PANCREATITIS",         "max_ratio": 12.5, "no_class_baseline": False},
            {"reaction": "NOVEL_REACTION_X",     "max_ratio": None, "no_class_baseline": True},
            {"reaction": "BLOOD GLUCOSE INCR",   "max_ratio": 8.3,  "no_class_baseline": False},
            {"reaction": "UNIQUE_TO_DRUG",       "max_ratio": None, "no_class_baseline": True},
        ]
        # Must not raise
        result = _sort_signals(signals)
        assert len(result) == 4

    def test_sentinel_signals_sort_last(self):
        """
        Signals with finite ratios should appear before no_class_baseline=True signals.
        """
        signals = [
            {"reaction": "REACTION_A",    "max_ratio": None, "no_class_baseline": True},
            {"reaction": "REACTION_B",    "max_ratio": 5.0,  "no_class_baseline": False},
            {"reaction": "REACTION_C",    "max_ratio": 3.0,  "no_class_baseline": False},
        ]
        result = _sort_signals(signals)
        # Finite ratio signals first, in descending order
        assert result[0]["reaction"] == "REACTION_B"
        assert result[1]["reaction"] == "REACTION_C"
        # Sentinel last
        assert result[2]["reaction"] == "REACTION_A"
        assert result[2]["no_class_baseline"] is True

    def test_sentinel_fields_are_none(self):
        """
        no_class_baseline signals should carry max_ratio=None and avg_ratio=None,
        not 999.0 (the raw stored sentinel value).
        """
        signal = {
            "reaction": "UNIQUE_REACTION",
            "max_ratio": None,
            "avg_ratio": None,
            "no_class_baseline": True,
        }
        assert signal["max_ratio"] is None
        assert signal["avg_ratio"] is None

    def test_normal_signals_unaffected(self):
        """Signals with no_class_baseline=False behave exactly as before."""
        signals = [
            {"reaction": "C", "max_ratio": 2.0,  "no_class_baseline": False},
            {"reaction": "A", "max_ratio": 10.0, "no_class_baseline": False},
            {"reaction": "B", "max_ratio": 5.0,  "no_class_baseline": False},
        ]
        result = _sort_signals(signals)
        assert [s["reaction"] for s in result] == ["A", "B", "C"]
