"""Executable statistical plan for H2 / H3 (protocol 7.9).

Input: one row per test episode (experiments/stats_pilot.py writes this table):
    mode ("A" frozen Baseline 1 + filter, "B" trained with the filter), D, rho, unit (training
    run in mode B, policy in mode A), unit_seed, episode_seed, v_eff, E, N_exec, N_blocked, N_att.

Covariates (fixed here, in code and in the protocol). D and rho are numeric and centred at the
means of the design grid, not of the data:
    Dc   = (D - 25) / 10        D in {10, 25, 40} veh/km/lane  ->  -1.5, 0, +1.5
    rhoc = rho - 2/3            rho in {0.4, 0.6, 1.0} s       ->  -0.267, -0.067, +0.333
    mode = 0 (A), 1 (B)
so beta_D and beta_rho are the slopes at the centre of the grid (the average slopes of a balanced
design) and beta_Dxrho does not depend on the centring; without centring the main-effect
coefficients would be slopes at D = 0 or rho = 0, outside the grid.

Models (analysis/models.py; u_unit ~ N(0, s_u^2) is the random intercept of the unit):
    M1  v_eff     = b0 + bD Dc + brho rhoc + bDxrho Dc rhoc + u_unit + e          LMM, mode B
    M2  N_exec    ~ NB2,  log mu = log E + b0 + bD Dc + brho rhoc + bDxrho Dc rhoc + u_unit,  mode B
    M3  N_blocked ~ NB2,  log mu = log E + b0 + bmode mode + bD Dc + brho rhoc + u_unit,  modes A and B
    M4  N_blocked ~ BetaBin(N_att, p, phi), logit p = b0 + bmode mode + bD Dc + brho rhoc + u_unit,
        modes A and B, episodes with N_att = 0 excluded: the final model of P(block | attempt)
    M5  N_att     ~ NB2 as M3 (control, outside the family)
Episodes with E = 0 are excluded from M2, M3, M5 (their number is reported).

Primary family: K = 7 one-sided checks, each a named function below (PRIMARY).
Primary inference, the same for all seven (LMM included; no Kenward-Roger): signed likelihood
root with a parametric bootstrap under the null model (models.bootstrap_signed_root), Holm over
the seven one-sided p-values.

Mandatory sensitivity analysis that keeps the shared scenario ids: two-way cluster bootstrap -
units are resampled inside their cell (mode B: (D, rho); mode A: D, a policy keeps all its rho)
and scenarios (D, episode_seed) inside D; a row enters with the product of the two
multiplicities, so a drawn scenario brings its episodes of every unit. The statistic is the
coefficient of a linear model on unit aggregates (mean v_eff; log of the pooled rate; logit of the
pooled blocking share). Its one-sided p-value is the bootstrap share of estimates on the wrong
side of zero; Holm over the same seven checks.

Reconciliation (reconcile): a check is "established" only if the primary model converged, both
Holm-adjusted p-values are <= alpha, and both estimates have the hypothesised sign. A model that
does not converge (optimiser, Hessian, quadrature or more than 10 % failed bootstrap fits) is an
explicit failure: its p-value enters Holm as 1, the family stays K = 7, the check is "not
established (model failure)" - there is no fallback model.
"""
from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy import special

from analysis.models import LMM, BetaBinGLMM, NB2GLMM, bootstrap_signed_root

D_CENTER, D_UNIT, RHO_CENTER = 25.0, 10.0, 2.0 / 3.0
ALPHA = 0.05
ESTABLISHED = "established"


def Dc(D):
    return (np.asarray(D, float) - D_CENTER) / D_UNIT


def rhoc(rho):
    return np.asarray(rho, float) - RHO_CENTER


TERMS = {"Intercept": lambda r: np.ones(len(r["D"])), "D": lambda r: Dc(r["D"]),
         "rho": lambda r: rhoc(r["rho"]), "D:rho": lambda r: Dc(r["D"]) * rhoc(r["rho"]),
         "mode": lambda r: (np.asarray(r["mode"]) == "B").astype(float)}
H2_TERMS = ("Intercept", "D", "rho", "D:rho")
H3_TERMS = ("Intercept", "mode", "D", "rho")


def columns(rows, keep=None):
    """Rows (dicts) -> dict of arrays; keep: boolean function of a row."""
    rows = [r for r in rows if keep is None or keep(r)]
    num = ("D", "rho", "v_eff", "E", "N_exec", "N_blocked", "N_att")
    out = {k: np.array([float(r[k]) for r in rows]) for k in num}
    for k in ("mode", "unit", "unit_seed", "episode_seed"):
        out[k] = np.array([str(r[k]) for r in rows])
    return out


def design(cols, terms):
    return np.column_stack([TERMS[t](cols) for t in terms]), list(terms)


# ---------------------------------------------------------------------- models M1 - M5
def model_veff(rows):
    c = columns(rows, lambda r: r["mode"] == "B")
    X, names = design(c, H2_TERMS)
    return LMM(c["v_eff"], X, names, c["unit"])


def model_nexec(rows):
    c = columns(rows, lambda r: r["mode"] == "B" and float(r["E"]) > 0)
    X, names = design(c, H2_TERMS)
    return NB2GLMM(c["N_exec"], X, names, c["unit"], offset=np.log(c["E"]))


def model_nblocked(rows):
    c = columns(rows, lambda r: r["mode"] in "AB" and float(r["E"]) > 0)
    X, names = design(c, H3_TERMS)
    return NB2GLMM(c["N_blocked"], X, names, c["unit"], offset=np.log(c["E"]))


def model_pblock(rows):
    c = columns(rows, lambda r: r["mode"] in "AB" and float(r["N_att"]) > 0)
    X, names = design(c, H3_TERMS)
    return BetaBinGLMM(c["N_blocked"], X, names, c["unit"], trials=c["N_att"])


def model_natt(rows):
    c = columns(rows, lambda r: r["mode"] in "AB" and float(r["E"]) > 0)
    X, names = design(c, H3_TERMS)
    return NB2GLMM(c["N_att"], X, names, c["unit"], offset=np.log(c["E"]))


# ---------------------------------------------------------------------- the seven primary checks
def h2_veff_rho(rows, n_boot, rng):
    """H2: v_eff decreases with rho (b_rho < 0 in M1)."""
    return bootstrap_signed_root(model_veff(rows), "rho", -1, n_boot, rng)


def h2_veff_D(rows, n_boot, rng):
    """H2: v_eff decreases with D (b_D < 0 in M1)."""
    return bootstrap_signed_root(model_veff(rows), "D", -1, n_boot, rng)


def h2_veff_Dxrho(rows, n_boot, rng):
    """H2: the loss of v_eff with rho is larger at high D (b_Dxrho < 0 in M1)."""
    return bootstrap_signed_root(model_veff(rows), "D:rho", -1, n_boot, rng)


def h2_nexec_rho(rows, n_boot, rng):
    """H2: the rate of executed lane changes decreases with rho (b_rho < 0 in M2)."""
    return bootstrap_signed_root(model_nexec(rows), "rho", -1, n_boot, rng)


def h2_nexec_D(rows, n_boot, rng):
    """H2: the rate of executed lane changes decreases with D (b_D < 0 in M2)."""
    return bootstrap_signed_root(model_nexec(rows), "D", -1, n_boot, rng)


def h3a_nblocked_mode(rows, n_boot, rng):
    """H3a: mode B has a lower rate of blocked attempts than mode A (b_mode < 0 in M3)."""
    return bootstrap_signed_root(model_nblocked(rows), "mode", -1, n_boot, rng)


def h3b_pblock_mode(rows, n_boot, rng):
    """H3b: mode B has a lower P(block | attempt) than mode A (b_mode < 0 in M4)."""
    return bootstrap_signed_root(model_pblock(rows), "mode", -1, n_boot, rng)


def control_natt_mode(rows, n_boot, rng):
    """Control (outside the family): rate of attempts, mode B against mode A (M5)."""
    return bootstrap_signed_root(model_natt(rows), "mode", -1, n_boot, rng)


@dataclass(frozen=True)
class Check:
    key: str
    hypothesis: str
    function: object
    model: str
    outcome: str
    term: str
    terms: tuple
    kind: str             # aggregate of the sensitivity analysis: mean / rate / share
    direction: int = -1


PRIMARY = (
    Check("H2: v_eff ~ rho", "H2", h2_veff_rho, "M1 LMM", "v_eff", "rho", H2_TERMS, "mean"),
    Check("H2: v_eff ~ D", "H2", h2_veff_D, "M1 LMM", "v_eff", "D", H2_TERMS, "mean"),
    Check("H2: v_eff ~ D x rho", "H2", h2_veff_Dxrho, "M1 LMM", "v_eff", "D:rho", H2_TERMS, "mean"),
    Check("H2: N_exec ~ rho", "H2", h2_nexec_rho, "M2 NB2-GLMM", "N_exec", "rho", H2_TERMS, "rate"),
    Check("H2: N_exec ~ D", "H2", h2_nexec_D, "M2 NB2-GLMM", "N_exec", "D", H2_TERMS, "rate"),
    Check("H3a: N_blocked ~ mode", "H3a", h3a_nblocked_mode, "M3 NB2-GLMM", "N_blocked", "mode", H3_TERMS, "rate"),
    Check("H3b: P(block ∣ attempt) ~ mode", "H3b", h3b_pblock_mode, "M4 beta-binomial GLMM", "N_blocked",
          "mode", H3_TERMS, "share"),
)
CONTROL = Check("control: N_att ~ mode", "control", control_natt_mode, "M5 NB2-GLMM", "N_att", "mode",
                H3_TERMS, "rate")


# ---------------------------------------------------------------------- Holm and reconciliation
def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (step-down); None counts as 1."""
    p = np.array([1.0 if v is None else float(v) for v in pvalues])
    order = np.argsort(p, kind="stable")
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def reconcile(check, primary, p_primary_holm, sens, p_sens_holm, alpha=ALPHA):
    """Outcome of one check from the two analyses (see the module docstring)."""
    if not primary["converged"]:
        return "not established (model failure)"
    if sens["estimate"] is None or sens["p"] is None:
        return "not established (sensitivity analysis failed)"
    if np.sign(primary["estimate"]) != check.direction or np.sign(sens["estimate"]) != check.direction:
        return "not established (direction)"
    if p_primary_holm > alpha:
        return "not established (primary analysis)"
    if p_sens_holm > alpha:
        return "not established (sensitivity analysis)"
    return ESTABLISHED


# ---------------------------------------------------------------------- sensitivity analysis
def _aggregate(c, w, kind, outcome):
    """Weighted unit-cell aggregates. Returns (keys, values, weights) or None if undefined."""
    keys = np.array([f"{u}|{d:g}|{r:g}" for u, d, r in zip(c["unit"], c["D"], c["rho"])])
    uniq, inv = np.unique(keys, return_inverse=True)
    def s(x):
        return np.bincount(inv, w * x, minlength=len(uniq))
    n = s(np.ones(len(w)))
    use = n > 0
    if kind == "mean":
        val = s(c["v_eff"])[use] / n[use]
    elif kind == "rate":
        e = s(c["E"])[use]
        if np.any(e <= 0):
            return None
        val = np.log((s(c[outcome])[use] + 0.5) / e)
    else:
        att = s(c["N_att"])[use]
        val = special.logit((s(c["N_blocked"])[use] + 0.5) / (att + 1.0))
    first = np.array([np.flatnonzero(inv == k)[0] for k in np.flatnonzero(use)])
    return first, val


def _wls(c, first, val, mult, terms, term):
    cols = {k: v[first] for k, v in c.items()}
    X, names = design(cols, terms)
    W = mult[first]
    A = X.T @ (X * W[:, None])
    if np.linalg.matrix_rank(A) < A.shape[0]:
        return None
    return float(np.linalg.solve(A, X.T @ (W * val))[names.index(term)])


def two_way_resampler(c):
    """Two-way cluster bootstrap that keeps the shared scenario ids. Returns draw(rng) ->
    (unit multiplicity, scenario multiplicity) per row: units are resampled inside their cell
    (mode B: (D, rho); other modes: D - a policy keeps all its rho), scenarios
    (D, episode_seed) inside D, so a drawn scenario brings its episodes of every unit."""
    stratum = np.array([f"{m}|{d:g}" + (f"|{r:g}" if m == "B" else "")
                        for m, d, r in zip(c["mode"], c["D"], c["rho"])])
    units = {s: np.unique(c["unit"][stratum == s]) for s in np.unique(stratum)}
    scen = {d: np.unique(c["episode_seed"][c["D"] == d]) for d in np.unique(c["D"])}

    def draw(rng):
        mu, ms = {}, {}
        for us in units.values():
            for u in rng.choice(us, len(us)):
                mu[u] = mu.get(u, 0) + 1
        for d, ss in scen.items():
            for x in rng.choice(ss, len(ss)):
                ms[(d, x)] = ms.get((d, x), 0) + 1
        wu = np.array([mu.get(u, 0) for u in c["unit"]], float)
        ws = np.array([ms.get((d, x), 0) for d, x in zip(c["D"], c["episode_seed"])], float)
        return wu, ws
    return draw


def sensitivity(rows, check, n_boot, rng, level=0.95):
    """Two-way cluster bootstrap (units x shared scenarios) of the aggregate linear model."""
    modes = "B" if check.hypothesis == "H2" else "AB"
    c = columns(rows, lambda r: r["mode"] in modes)
    ones = np.ones(len(c["D"]))
    agg = _aggregate(c, ones, check.kind, check.outcome)
    est = None if agg is None else _wls(c, agg[0], agg[1], ones, check.terms, check.term)
    draw = two_way_resampler(c)
    draws, failed = [], 0
    for _ in range(n_boot):
        wu, ws = draw(rng)
        a = _aggregate(c, wu * ws, check.kind, check.outcome)
        b = None if a is None else _wls(c, a[0], a[1], wu, check.terms, check.term)
        if b is None or not np.isfinite(b):
            failed += 1
        else:
            draws.append(b)
    out = dict(estimate=est, p=None, lower=None, upper=None, n_boot=n_boot, n_failed=failed)
    if est is not None and draws and failed <= 0.1 * n_boot:
        d = np.asarray(draws)
        out.update(p=float((1 + np.sum(check.direction * d <= 0)) / (len(d) + 1)),
                   lower=float(np.quantile(d, (1 - level) / 2)), upper=float(np.quantile(d, 1 - (1 - level) / 2)))
    return out


def run_primary(rows, n_boot, n_boot_sens, seed):
    """All seven checks, both analyses, Holm in each, reconciliation. Returns a list of dicts."""
    res = []
    for i, ch in enumerate(PRIMARY):
        prim = ch.function(rows, n_boot, np.random.default_rng([seed, i]))
        sens = sensitivity(rows, ch, n_boot_sens, np.random.default_rng([seed, 100 + i]))
        res.append(dict(check=ch.key, hypothesis=ch.hypothesis, function=ch.function.__name__,
                        model=ch.model, term=ch.term, primary=prim, sensitivity=sens))
    hp = holm([r["primary"]["p_boot"] if r["primary"]["converged"] else None for r in res])
    hs = holm([r["sensitivity"]["p"] for r in res])
    for ch, r, a, b in zip(PRIMARY, res, hp, hs):
        r.update(p_holm_primary=a, p_holm_sensitivity=b,
                 outcome=reconcile(ch, r["primary"], a, r["sensitivity"], b))
    return res


# ---------------------------------------------------------------------- descriptive and secondary
def cell_means(c, w=None, outcome="v_eff"):
    """{(D, rho): weighted mean of the outcome (v_eff) or pooled rate (counts per E)}."""
    w = np.ones(len(c["D"])) if w is None else w
    out = {}
    for d, r in sorted(set(zip(c["D"], c["rho"]))):
        sel = (c["D"] == d) & (c["rho"] == r)
        den = w[sel].sum() if outcome == "v_eff" else (w[sel] * c["E"][sel]).sum()
        out[(d, r)] = float((w[sel] * c[outcome][sel]).sum() / den) if den > 0 else float("nan")
    return out


def bootstrap_cells(rows, outcome, n_boot, rng, modes="B"):
    """Cell means of the data and of n_boot two-way cluster bootstrap replicates."""
    c = columns(rows, lambda r: r["mode"] in modes)
    draw = two_way_resampler(c)
    reps = []
    for _ in range(n_boot):
        wu, ws = draw(rng)
        reps.append(cell_means(c, wu * ws, outcome))
    return cell_means(c, None, outcome), reps


def interval(values, level=0.95):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return None, None
    return float(np.quantile(v, (1 - level) / 2)), float(np.quantile(v, 1 - (1 - level) / 2))


def contrast_of_differences(means, d_hi=40.0, d_lo=10.0, r_hi=1.0, r_lo=0.4):
    """[v(d_hi, r_hi) - v(d_hi, r_lo)] - [v(d_lo, r_hi) - v(d_lo, r_lo)] from cell means."""
    return (means[(d_hi, r_hi)] - means[(d_hi, r_lo)]) - (means[(d_lo, r_hi)] - means[(d_lo, r_lo)])


def adjacent_contrasts(means):
    """Differences between neighbouring levels: rho 0.4 -> 0.6 -> 1.0 and D 10 -> 25 -> 40."""
    Ds, Rs = sorted({d for d, _ in means}), sorted({r for _, r in means})
    out = {}
    for a, b in zip(Rs, Rs[1:]):
        out[f"rho {a:g} -> {b:g}"] = float(np.mean([means[(d, b)] - means[(d, a)] for d in Ds]))
    for a, b in zip(Ds, Ds[1:]):
        out[f"D {a:g} -> {b:g}"] = float(np.mean([means[(b, r)] - means[(a, r)] for r in Rs]))
    return out


def page_test(table):
    """Page's trend test for a decreasing response over ordered treatments (columns) in blocks
    (rows). L = sum_j j * (rank sum of column j) with ranks of the negated values; the p-value is
    exact (all rank assignments) when there are at most 50 000 of them, else normal."""
    t = -np.asarray(table, float)
    n, k = t.shape
    ranks = np.argsort(np.argsort(t, axis=1), axis=1) + 1.0
    L = float((ranks.sum(axis=0) * np.arange(1, k + 1)).sum())
    perms = list(permutations(range(1, k + 1)))
    if len(perms) ** n <= 50_000:
        per_block = [sum(j * p[j - 1] for j in range(1, k + 1)) for p in perms]
        dist = np.array([sum(c) for c in product(per_block, repeat=n)], float)
        return L, float(np.mean(dist >= L - 1e-9)), "exact"
    mean = n * k * (k + 1) ** 2 / 4.0
    var = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144.0
    return L, float(special.ndtr(-(L - mean) / np.sqrt(var))), "normal"


def trend_table(rows, outcome, over, modes="B"):
    """Blocks x ordered levels of `over` ("rho" or "D"): unit means; a block is the other factor
    and the unit seed. Incomplete blocks are dropped."""
    c = columns(rows, lambda r: r["mode"] in modes)
    other = "D" if over == "rho" else "rho"
    levels = sorted(set(c[over]))
    table = []
    for o, s in sorted(set(zip(c[other], c["unit_seed"]))):
        row = []
        for lv in levels:
            sel = (c[other] == o) & (c["unit_seed"] == s) & (c[over] == lv)
            if not sel.any():
                break
            row.append(c[outcome][sel].sum() / c["E"][sel].sum() if outcome != "v_eff" else c[outcome][sel].mean())
        else:
            table.append(row)
    return levels, np.array(table)


def dispersion(rows, outcome, terms, modes, trials=None):
    """Pearson dispersion of the model without the random effect and without overdispersion
    (Poisson with offset log E, or binomial): values well above 1 call for NB2 / beta-binomial."""
    need = "N_att" if trials else "E"
    c = columns(rows, lambda r: r["mode"] in modes and float(r[need]) > 0)
    X, _ = design(c, terms)
    y = c[outcome]
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        if trials:
            p = np.clip(special.expit(eta), 1e-9, 1 - 1e-9)
            mu, v = c[trials] * p, c[trials] * p * (1 - p)
        else:
            mu = c["E"] * np.exp(np.clip(eta, -30, 30))
            v = mu
        step = np.linalg.lstsq(X.T @ (X * v[:, None]), X.T @ (y - mu), rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    df = len(y) - X.shape[1]
    return float(np.sum((y - mu) ** 2 / np.maximum(v, 1e-12)) / df) if df > 0 else float("nan")
