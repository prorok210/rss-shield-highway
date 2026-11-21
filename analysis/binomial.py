"""Binomial part of the statistical plan (protocol 7.9): H1 decisions with exact Clopper-Pearson
bounds, the thresholds they imply, single-check power, bounds for a family of checks, and the
two-level cluster bootstrap of the descriptive difference between arms."""
import numpy as np
from scipy import stats

DELTA = 0.03       # risk threshold of H1
ALPHA = 0.05       # one-sided level of every bound

CONFIRMED, EXCEEDED, INSUFFICIENT = "confirmed", "exceeded", "insufficient"


def cp_upper(x, n, alpha=ALPHA):
    """Upper one-sided (1 - alpha) Clopper-Pearson bound for x events in n episodes."""
    return 1.0 if x >= n else float(stats.beta.ppf(1.0 - alpha, x + 1, n - x))


def cp_lower(x, n, alpha=ALPHA):
    """Lower one-sided (1 - alpha) Clopper-Pearson bound."""
    return 0.0 if x <= 0 else float(stats.beta.ppf(alpha, x, n - x + 1))


def h1_decision(x, n, delta=DELTA, alpha=ALPHA):
    """Three outcomes of one check: the bound is confirmed (upper bound <= delta), exceeded
    (lower bound > delta) or the data are insufficient. n = episodes with a correct start."""
    if n <= 0:
        return INSUFFICIENT
    if cp_upper(x, n, alpha) <= delta:
        return CONFIRMED
    if cp_lower(x, n, alpha) > delta:
        return EXCEEDED
    return INSUFFICIENT


def confirm_threshold(n, delta=DELTA, alpha=ALPHA):
    """Largest number of events with which the bound is confirmed (-1: impossible at this n)."""
    x = -1
    while x + 1 <= n and cp_upper(x + 1, n, alpha) <= delta:
        x += 1
    return x


def exceed_threshold(n, delta=DELTA, alpha=ALPHA):
    """Smallest number of events with which the exceedance is established (n + 1: impossible)."""
    x = 0
    while x <= n and cp_lower(x, n, alpha) <= delta:
        x += 1
    return x


def power_confirm(n, risk, delta=DELTA, alpha=ALPHA):
    """P(confirmed) of one check at the true risk."""
    c = confirm_threshold(n, delta, alpha)
    return float(stats.binom.cdf(c, n, risk)) if c >= 0 else 0.0


def power_exceed(n, risk, delta=DELTA, alpha=ALPHA):
    """P(exceedance established) of one check at the true risk."""
    return float(stats.binom.sf(exceed_threshold(n, delta, alpha) - 1, n, risk))


def boole_lower_bound(powers):
    """Lower bound of P(all checks succeed) that assumes nothing about their dependence:
    1 - sum(1 - p_i), not below 0 (Boole's inequality)."""
    return max(0.0, 1.0 - float(np.sum(1.0 - np.asarray(powers, float))))


def independent_product(powers):
    """P(all checks succeed) if the checks were independent. With shared scenarios they are not:
    this is a naive approximation, not the joint power."""
    return float(np.prod(np.asarray(powers, float)))


def bootstrap_h1_diff_twoway(prop_events_by_unit, b2_events_by_unit, *, n_boot=10_000,
                             seed=42, level=0.95):
    """Two-way cluster bootstrap of CR_ego_op(Proposed) - CR_ego_op(Baseline 2).

    The two arms share the episode_seed of every scenario, so resampling episodes independently
    inside each arm destroys the pairing and produces a non-zero interval even when both arms
    have identical event vectors.  Here the trained units are resampled independently inside each
    arm, while the common correct-start scenarios are resampled with one integer multiplicity
    vector shared by both arms.  Scenario ids are kept, and the sums are accumulated in int64, so
    identical vectors give exactly diff = 0 and CI = [0, 0].  Extra Baseline 2 scenarios (used only
    for H1b power) are excluded from the paired difference and counted explicitly."""
    p_units = sorted(prop_events_by_unit)
    b_units = sorted(b2_events_by_unit)
    if not p_units or not b_units:
        raise ValueError("both arms must have at least one trained unit")

    p_sets = [set(prop_events_by_unit[u]) for u in p_units]
    b_sets = [set(b2_events_by_unit[v]) for v in b_units]
    if any(s != p_sets[0] for s in p_sets[1:]):
        raise ValueError("inconsistent valid scenario ids across Proposed units")
    if any(s != b_sets[0] for s in b_sets[1:]):
        raise ValueError("inconsistent valid scenario ids across Baseline 2 units")

    s_prop, s_b2 = p_sets[0], b_sets[0]
    s_common = sorted(s_prop & s_b2)
    if not s_common:
        raise ValueError("no common valid scenarios between Proposed and Baseline 2")

    Yp = np.array([[int(prop_events_by_unit[u][s]) for s in s_common] for u in p_units],
                  dtype=np.int64)
    Yb = np.array([[int(b2_events_by_unit[v][s]) for s in s_common] for v in b_units],
                  dtype=np.int64)
    Mp, ns = Yp.shape
    Mb = Yb.shape[0]
    denom_p, denom_b = float(Mp * ns), float(Mb * ns)
    diff_hat = float(Yp.sum() / denom_p - Yb.sum() / denom_b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    chunk = 1000
    for start in range(0, n_boot, chunk):
        end = min(start + chunk, n_boot)
        nb = end - start

        idx_s = rng.integers(0, ns, size=(nb, ns))
        m_s = np.zeros((nb, ns), dtype=np.int64)
        np.add.at(m_s, (np.repeat(np.arange(nb), ns), idx_s.ravel()), 1)

        idx_p = rng.integers(0, Mp, size=(nb, Mp))
        w_p = np.zeros((nb, Mp), dtype=np.int64)
        np.add.at(w_p, (np.repeat(np.arange(nb), Mp), idx_p.ravel()), 1)

        idx_b = rng.integers(0, Mb, size=(nb, Mb))
        w_b = np.zeros((nb, Mb), dtype=np.int64)
        np.add.at(w_b, (np.repeat(np.arange(nb), Mb), idx_b.ravel()), 1)

        num_p = np.sum(w_p * (Yp @ m_s.T).T, axis=1)
        num_b = np.sum(w_b * (Yb @ m_s.T).T, axis=1)
        diffs[start:end] = num_p / denom_p - num_b / denom_b

    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"diff": diff_hat, "ci_low": float(lo), "ci_high": float(hi),
            "n_common_scenarios": ns, "n_extra_b2_excluded": len(s_b2 - s_prop),
            "n_extra_prop_excluded": len(s_prop - s_b2),
            "n_units_proposed": Mp, "n_units_baseline2": Mb}
