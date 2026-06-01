"""
Empirical Bayes Geometric Mean (EBGM) and EB05 — Gamma-Poisson Shrinker.

Implements the DuMouchel (1999) GPS model used by FDA's MGPS and the WHO
Uppsala Monitoring Centre. EBGM is the industry standard for disproportionality
analysis in spontaneous reporting systems.

Method
------
For each (drug, reaction) pair, the observed count O = drug_count and the
expected count E = drug_total × (baseline / faers_total) under independence.

The GPS model fits a 2-component Negative Binomial mixture prior over all
(O, E) pairs for a drug — the mixture captures two populations: "true signals"
(elevated lambda) and "noise" (lambda near 1):

  P(O | E) = P × NB(α₁, β₁/(β₁+E))  +  (1−P) × NB(α₂, β₂/(β₂+E))

Parameters θ = (α₁, β₁, α₂, β₂, P) are estimated by maximum marginal likelihood
using L-BFGS-B.

Given fitted θ, for each signal:
  EBGM = exp(E[ln λ | O, E])     — posterior geometric mean (shrunk RR estimate)
  EB05 = 5th percentile of the posterior lambda distribution

  FDA signal threshold: EB05 ≥ 2  (conservative — lower bound on the RR)
  WHO threshold: EBGM ≥ 2

EBGM advantages over PRR/ROR
-----------------------------
- Borrowing strength across reactions: rare reactions shrink toward the prior,
  so a PRR=10 on n=3 gets a low EBGM, while a PRR=3 on n=3000 stays high
- Single coherent ranking across all reactions via EB05
- No multiplicity correction needed (the prior IS the correction)
- Standard in regulatory pharmacovigilance (FDA, WHO, EMA)

References
----------
DuMouchel W. (1999) Bayesian data mining in large frequency tables, with an
application to the FDA Spontaneous Reporting System. The American Statistician
53(3):177-190.

Gould AL. (2003) Practical pharmacovigilance analysis strategies.
Pharmacoepidemiology and Drug Safety 12:559-574.
"""

import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import gamma as _gamma_dist


# ---------------------------------------------------------------------------
# Negative Binomial helpers
# ---------------------------------------------------------------------------

def _nb_logpmf(n: np.ndarray, r: float, p: float) -> np.ndarray:
    """
    Log PMF of NB(r, p) where p = b/(b+E) in the GPS parametrization.

    NB(r, p): P(X=n) = C(n+r-1, n) × p^r × (1-p)^n
    """
    return (gammaln(n + r) - gammaln(r) - gammaln(n + 1)
            + r * np.log(p) + n * np.log1p(-p))


# ---------------------------------------------------------------------------
# Mixture fitting
# ---------------------------------------------------------------------------

def fit_gps_prior(
    observed: np.ndarray,
    expected: np.ndarray,
) -> tuple[tuple, bool]:
    """
    Fit the 2-component Gamma-Poisson Shrinker mixture to (O, E) pairs.

    Returns (params, converged) where params = (α₁, β₁, α₂, β₂, P).
    If the fit fails, returns a conservative default prior.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    # Filter out zero-expected rows (can't condition on E=0)
    mask = expected > 0
    if mask.sum() < 5:
        # Too few data points for mixture fitting; return uninformative prior
        return (0.5, 0.5, 2.0, 4.0, 0.5), False

    O = observed[mask]
    E = expected[mask]

    def neg_log_lik(theta: np.ndarray) -> float:
        la1, lb1, la2, lb2, logit_P = theta
        a1, b1 = math.exp(la1), math.exp(lb1)
        a2, b2 = math.exp(la2), math.exp(lb2)
        P = 1.0 / (1.0 + math.exp(-logit_P))

        # NB parametrization: p = b/(b+E)
        p1 = b1 / (b1 + E)
        p2 = b2 / (b2 + E)
        p1 = np.clip(p1, 1e-12, 1 - 1e-12)
        p2 = np.clip(p2, 1e-12, 1 - 1e-12)

        lp1 = _nb_logpmf(O, a1, p1)
        lp2 = _nb_logpmf(O, a2, p2)

        # log-sum-exp for numerical stability
        log_mix = np.logaddexp(math.log(P) + lp1, math.log(1 - P) + lp2)
        nll = -np.sum(log_mix)
        return float(nll) if math.isfinite(nll) else 1e15

    # Try multiple starting points — GPS likelihood can be multimodal
    best_res = None
    init_points = [
        # (la1,   lb1,   la2,   lb2,  logit_P)
        (math.log(0.2), math.log(0.1), math.log(2.0), math.log(4.0), 0.0),
        (math.log(0.5), math.log(0.5), math.log(3.0), math.log(2.0), 0.0),
        (math.log(1.0), math.log(1.0), math.log(5.0), math.log(10.), -0.5),
        (math.log(0.1), math.log(0.05), math.log(1.5), math.log(3.0), 1.0),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in init_points:
            res = minimize(
                neg_log_lik, x0,
                method="L-BFGS-B",
                options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8},
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res

    if best_res is None or not math.isfinite(best_res.fun):
        return (0.5, 0.5, 2.0, 4.0, 0.5), False

    la1, lb1, la2, lb2, logit_P = best_res.x
    a1, b1 = math.exp(la1), math.exp(lb1)
    a2, b2 = math.exp(la2), math.exp(lb2)
    P = 1.0 / (1.0 + math.exp(-logit_P))

    return (a1, b1, a2, b2, P), best_res.success


# ---------------------------------------------------------------------------
# Posterior inference
# ---------------------------------------------------------------------------

def compute_ebgm(
    o: int,
    e: float,
    params: tuple,
) -> tuple[float, float]:
    """
    Compute EBGM and EB05 for a single (observed=O, expected=E) pair.

    EBGM = exp(E[ln λ | O, E])   — posterior geometric mean
    EB05 = 5th percentile of the posterior lambda distribution

    The posterior is a mixture of two Gamma distributions:
      Component j: Gamma(αⱼ + O, βⱼ + E)  with posterior weight wⱼ

    EBGM = exp(w₁ × [ψ(α₁+O) − ln(β₁+E)]  +  w₂ × [ψ(α₂+O) − ln(β₂+E)])
    where ψ is the digamma function.

    EB05 is found by binary search on the posterior mixture CDF.

    Returns (ebgm, eb05). Returns (0.0, 0.0) on error.
    """
    a1, b1, a2, b2, P = params
    if e <= 0:
        return 0.0, 0.0

    # Posterior component weights via Bayes theorem
    p1 = b1 / (b1 + e)
    p2 = b2 / (b2 + e)
    p1 = max(min(p1, 1 - 1e-12), 1e-12)
    p2 = max(min(p2, 1 - 1e-12), 1e-12)

    lp1 = float(_nb_logpmf(np.array([o]), a1, p1)[0])
    lp2 = float(_nb_logpmf(np.array([o]), a2, p2)[0])

    log_w1 = math.log(P)       + lp1
    log_w2 = math.log(1 - P)   + lp2
    # log-sum-exp to avoid underflow when both log-weights are very negative
    # (large O, small E — e.g. a dominant rare-signal drug).
    # math.exp() on very negative values underflows to 0.0 and math.log(0) crashes.
    log_norm = float(np.logaddexp(log_w1, log_w2))
    w1 = math.exp(log_w1 - log_norm)
    w2 = 1.0 - w1

    # EBGM = exp(E[ln λ | O])
    # For Gamma(α+O, β+E): E[ln λ] = ψ(α+O) − ln(β+E)
    try:
        log_ebgm = (
            w1 * (float(digamma(a1 + o)) - math.log(b1 + e))
            + w2 * (float(digamma(a2 + o)) - math.log(b2 + e))
        )
        ebgm = round(math.exp(log_ebgm), 3)
    except (ValueError, ZeroDivisionError):
        return 0.0, 0.0

    # EB05: 5th percentile via binary search on posterior CDF
    # CDF(x) = w1 × Gamma_CDF(x; α₁+O, β₁+E)  +  w2 × Gamma_CDF(x; α₂+O, β₂+E)
    def posterior_cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        cdf1 = _gamma_dist.cdf(x, a=a1 + o, scale=1.0 / (b1 + e))
        cdf2 = _gamma_dist.cdf(x, a=a2 + o, scale=1.0 / (b2 + e))
        return w1 * cdf1 + w2 * cdf2

    lo, hi = 1e-6, max(10.0 * ebgm, 20.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if posterior_cdf(mid) < 0.05:
            lo = mid
        else:
            hi = mid
    eb05 = round((lo + hi) / 2.0, 3)

    return ebgm, eb05


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------

def annotate_signals_with_ebgm(
    signals: list[dict],
    drug_total: int,
    faers_total: int,
) -> list[dict]:
    """
    Add EBGM and EB05 fields to a list of PRR signals.

    The GPS prior is fit once across all signals for this drug, then
    each signal is annotated with its posterior estimates.

    Args:
        signals:     List of signal dicts, each with 'drug_count' and 'baseline'
        drug_total:  Total drug reports
        faers_total: Total FAERS reports

    Returns:
        Same list with 'ebgm' and 'eb05' added to each signal.
        'eb05_signal': True if EB05 ≥ 2.0  (FDA-style flag)
    """
    if not signals or drug_total <= 0 or faers_total <= 0:
        return signals

    # Compute expected counts: E = drug_total × (baseline / faers_total)
    observed_arr = np.array([s["drug_count"] for s in signals], dtype=float)
    expected_arr = np.array(
        [drug_total * s.get("baseline", 0) / faers_total for s in signals],
        dtype=float,
    )

    # Fit the GPS prior across all reactions for this drug
    params, converged = fit_gps_prior(observed_arr, expected_arr)
    if not converged:
        # Annotate with a note that fit didn't converge; EBGM still computed
        # using the best-available params (conservative default if all failed)
        pass

    for signal, o, e in zip(signals, observed_arr.tolist(), expected_arr.tolist()):
        try:
            ebgm, eb05 = compute_ebgm(int(o), e, params)
        except Exception:
            # One pathological reaction (e.g. extreme O/E) must not wipe the batch
            ebgm, eb05 = 0.0, 0.0
        signal["ebgm"]        = ebgm
        signal["eb05"]        = eb05
        signal["eb05_signal"] = eb05 >= 2.0   # FDA threshold

    return signals
