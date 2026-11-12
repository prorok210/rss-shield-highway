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


def cluster_bootstrap_difference(a, b, n_boot, rng, level=0.95):
    """Difference of event shares mean(a) - mean(b) between two groups of policies with a
    two-level cluster bootstrap: policies are resampled with replacement inside each group, then
    episodes inside each drawn policy. a, b: lists of 0/1 arrays, one per policy.
    Returns (difference, lower, upper) with a percentile interval."""
    def share(group):
        return float(np.mean([np.mean(x) for x in group]))

    def draw(group):
        out = []
        for i in rng.integers(0, len(group), len(group)):
            x = group[i]
            out.append(np.mean(x[rng.integers(0, len(x), len(x))]))
        return float(np.mean(out))

    a = [np.asarray(x, float) for x in a]
    b = [np.asarray(x, float) for x in b]
    diffs = np.array([draw(a) - draw(b) for _ in range(n_boot)])
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return share(a) - share(b), float(lo), float(hi)
