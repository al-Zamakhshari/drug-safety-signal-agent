"""
BCPNN — Bayesian Confidence Propagation Neural Network.

The WHO Uppsala Monitoring Centre's Bayesian disproportionality estimator,
published alongside EBGM as the standard alternative. Both answer the same
question (is this drug-reaction pair more frequent than expected after Bayesian
shrinkage?), via different prior assumptions:

  GPS/EBGM (DuMouchel 1999):  Gamma-Poisson  — models the *count* directly
  BCPNN   (Bate 1998):        Beta-Binomial   — models the *rate* (probability)

They typically agree closely; disagreements at small n are informative.

Information Component (IC)
--------------------------
IC = log₂(p₁₁ / (p₁. × p.₁))

  The log₂ of the observed co-occurrence probability relative to the probability
  expected under independence. IC > 0 means disproportionate reporting.

Norén (2006) shrinkage estimate (closed-form, replaces original Gibbs sampler):
  E    = (n1. × n.1) / N          (expected count under independence)
  IC   = log₂((n11 + 0.5) / (E + 0.5))

Bate (1998) closed-form variance of IC:
  γ    = (N+2)² / ((n1. + 2)(n.1 + 2))        (adaptive prior strength)
  V(IC) = (1/ln2)² × [
            (N − n11 + γ − 1) / ((n11 + 1)(1 + N + γ))
          + (N − n1. + 1)     / ((n1. + 1)(1 + N + 2))
          + (N − n.1 + 1)     / ((n.1 + 1)(1 + N + 2))
          ]
  IC_SD = √V(IC)

Signal thresholds:
  IC025 = IC − 2·IC_SD       (≈ 2.5th percentile of posterior)
  IC975 = IC + 2·IC_SD
  ic_signal = IC025 > 0      — WHO UMC standard signal flag

Notation mapped to the 2×2 table in prr.py:
  n11   = a  = drug_count              (drug AND reaction)
  n1.   = a+b = drug_total             (all drug reports)
  n.1   = a+c = baseline               (all reports with this reaction)
  N          = faers_total             (all FAERS reports)

References:
  Bate A et al. (1998) A Bayesian neural network method for adverse drug
  reaction signal generation. Eur J Clin Pharmacol 54(4):315-321.
  Norén GN et al. (2006) Shrinkage observed-to-expected ratios for robust
  and transparent large-scale pattern discovery. Statistical Methods in
  Medical Research 15(1):3-16.
  WHO UMC. Vigibase signal detection methodology (ic.who-umc.org).
"""

import math


def compute_ic(
    n11: int,
    n1_dot: int,
    n_dot1: int,
    N: int,
) -> tuple[float, float, float]:
    """
    Compute IC, IC025, IC975 for one drug × reaction cell.

    Args:
        n11:    drug_count       — reports with both drug AND reaction
        n1_dot: drug_total       — all reports mentioning the drug
        n_dot1: baseline         — all reports mentioning the reaction
        N:      faers_total      — all FAERS reports

    Returns:
        (ic, ic025, ic975)
        ic_signal = ic025 > 0.0   (compute at the call site)
    """
    if N <= 0 or n1_dot <= 0 or n_dot1 <= 0:
        return 0.0, 0.0, 0.0

    # Expected count under independence
    E = (n1_dot * n_dot1) / N

    # IC point estimate (Norén 2006 shrinkage)
    ic = math.log2((n11 + 0.5) / (E + 0.5))

    # Adaptive prior strength (Bate 1998)
    gamma = ((N + 2) ** 2) / ((n1_dot + 2) * (n_dot1 + 2))

    # Variance terms (priors: α₁ = β₁ = 1, α = β = 2, γ₁₁ = 1)
    inv_ln2_sq = (1.0 / math.log(2.0)) ** 2
    term1 = (N - n11 + gamma - 1.0) / ((n11 + 1.0) * (1.0 + N + gamma))
    term2 = (N - n1_dot + 1.0)      / ((n1_dot + 1.0) * (1.0 + N + 2.0))
    term3 = (N - n_dot1 + 1.0)      / ((n_dot1 + 1.0) * (1.0 + N + 2.0))

    var_ic = inv_ln2_sq * (term1 + term2 + term3)
    ic_sd  = math.sqrt(max(var_ic, 0.0))

    ic025 = ic - 2.0 * ic_sd
    ic975 = ic + 2.0 * ic_sd

    return round(ic, 3), round(ic025, 3), round(ic975, 3)


def annotate_signals_with_bcpnn(
    signals: list[dict],
    drug_total: int,
    faers_total: int,
) -> list[dict]:
    """
    Add IC, IC025, IC975, ic_signal fields to a list of PRR signals.

    Uses the same inputs already present on each signal dict from calculate_prr:
      drug_count  → n11
      drug_total  → n1.  (passed in)
      baseline    → n.1  (total reports with the reaction)
      faers_total → N    (passed in)

    Args:
        signals:     List of signal dicts from calculate_prr
        drug_total:  Total drug reports
        faers_total: Total FAERS reports

    Returns:
        Same list with 'ic', 'ic025', 'ic975', 'ic_signal' added.
        ic_signal: True when IC025 > 0 (WHO UMC threshold)
    """
    if not signals or drug_total <= 0 or faers_total <= 0:
        return signals

    for signal in signals:
        try:
            ic, ic025, ic975 = compute_ic(
                n11    = signal["drug_count"],
                n1_dot = drug_total,
                n_dot1 = signal.get("baseline", 0),
                N      = faers_total,
            )
            signal["ic"]        = ic
            signal["ic025"]     = ic025
            signal["ic975"]     = ic975
            signal["ic_signal"] = ic025 > 0.0   # WHO UMC signal threshold
        except Exception:
            # One bad signal must not wipe the batch
            signal["ic"]        = 0.0
            signal["ic025"]     = 0.0
            signal["ic975"]     = 0.0
            signal["ic_signal"] = False

    return signals
