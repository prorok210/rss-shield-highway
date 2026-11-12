"""Block 6: the statistical plan as an executable specification."""
import csv

import numpy as np
import pytest

from analysis import plan as P
from analysis import run as R
from analysis.models import LMM, BetaBinGLMM, NB2GLMM, bootstrap_signed_root, signed_root


def episodes(seed=0, units=3, n_ep=6, effect=1.0):
    """Synthetic episode table with the structure of the stats pilot and known negative effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for D in (10, 25, 40):
        for us in range(units):
            ua = rng.normal(0, 0.15)
            for k, rho in enumerate((0.4, 0.6, 1.0)):
                ub = rng.normal(0, 0.15)
                for mode, u, unit in (("A", ua, f"A:{D}:{us}"), ("B", ub, f"B:{D}:{rho}:{us}")):
                    for s in range(n_ep):
                        dc, rc, m = float(P.Dc(D)), float(P.rhoc(rho)), float(mode == "B")
                        E = int(rng.integers(120, 200))
                        att = rng.poisson(E * np.exp(-2.0 + u))
                        pb = 1 / (1 + np.exp(-(0.3 - effect * 1.2 * m + 0.4 * dc + 0.5 * rc + u)))
                        rows.append(dict(
                            mode=mode, D=D, rho=rho, unit=unit, unit_seed=us, episode_seed=900300 + s,
                            v_eff=18 - effect * (2.0 * dc + 6.0 * rc + 3.0 * dc * rc) + u + rng.normal(0, 0.5),
                            E=E, N_exec=rng.poisson(E * np.exp(-3.0 - effect * (0.5 * dc + 1.5 * rc) + u)),
                            N_att=att, N_blocked=rng.binomial(att, pb),
                            correct_start=1, rho_cls=rho, crashed=0, crash_category="", h1_event=0))
    return rows


# ---------------------------------------------------------------------- specification
def test_covariates_are_centred_at_the_design_grid():
    assert list(P.Dc([10, 25, 40])) == [-1.5, 0.0, 1.5]
    assert np.mean(P.rhoc([0.4, 0.6, 1.0])) == pytest.approx(0.0, abs=1e-12)
    cols = P.columns(episodes(units=1, n_ep=1))
    X, names = P.design(cols, P.H2_TERMS)
    assert names == ["Intercept", "D", "rho", "D:rho"]
    b = cols["mode"] == "B"
    assert X[b, 1].mean() == pytest.approx(0) and X[b, 2].mean() == pytest.approx(0)
    assert np.allclose(X[:, 3], X[:, 1] * X[:, 2])


def test_seven_primary_checks_are_named_functions_with_a_hypothesis():
    assert len(P.PRIMARY) == 7
    names = [c.function.__name__ for c in P.PRIMARY]
    assert names == ["h2_veff_rho", "h2_veff_D", "h2_veff_Dxrho", "h2_nexec_rho", "h2_nexec_D",
                     "h3a_nblocked_mode", "h3b_pblock_mode"]
    assert [c.hypothesis for c in P.PRIMARY] == ["H2"] * 5 + ["H3a", "H3b"]
    for c in P.PRIMARY:
        assert getattr(P, c.function.__name__) is c.function and c.function.__doc__.startswith(c.hypothesis)
        assert c.direction == -1 and c.term in c.terms
    assert P.CONTROL.function is P.control_natt_mode and P.CONTROL not in P.PRIMARY


def test_holm_and_the_reconciliation_rule():
    assert P.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert P.holm([0.2, None, 0.01]) == pytest.approx([0.4, 1.0, 0.03])        # a failed model counts as p = 1
    ch = P.PRIMARY[0]
    ok = dict(converged=True, estimate=-1.0)
    sens = dict(estimate=-0.8, p=0.01)
    assert P.reconcile(ch, ok, 0.01, sens, 0.02) == P.ESTABLISHED
    assert P.reconcile(ch, ok, 0.06, sens, 0.02) == "not established (primary analysis)"
    assert P.reconcile(ch, ok, 0.01, sens, 0.20) == "not established (sensitivity analysis)"
    assert P.reconcile(ch, ok, 0.01, dict(estimate=0.3, p=0.9), 1.0) == "not established (direction)"
    assert P.reconcile(ch, dict(converged=True, estimate=0.5), 0.9, sens, 0.02) == "not established (direction)"
    assert P.reconcile(ch, dict(converged=False, estimate=-1.0), 1.0, sens, 0.02) == "not established (model failure)"
    assert P.reconcile(ch, ok, 0.01, dict(estimate=None, p=None), 1.0) == "not established (sensitivity analysis failed)"


# ---------------------------------------------------------------------- models
def test_lmm_equals_the_closed_form_for_balanced_data():
    rows = [r for r in episodes(seed=1) if r["mode"] == "B"]
    m = P.model_veff(rows)
    m = m.with_y(m.y + np.random.default_rng(7).normal(0, 1.0, m.G)[m.g])   # clear unit effect
    f = m.fit()
    ols = np.linalg.lstsq(m.X, m.y, rcond=None)[0]              # balanced: GLS = OLS
    assert f.converged and np.allclose(f.beta, ols, atol=1e-6)
    # the reported log-likelihood is the dense Gaussian one, and it is maximal in the variances
    def dense(s_u, s_e):
        Z = (m.g[:, None] == np.arange(m.G)[None, :]).astype(float)
        V = s_e ** 2 * np.eye(len(m.y)) + s_u ** 2 * Z @ Z.T
        r = m.y - m.X @ f.beta
        return -0.5 * (len(r) * np.log(2 * np.pi) + np.linalg.slogdet(V)[1] + r @ np.linalg.solve(V, r))
    assert f.loglik == pytest.approx(dense(f.sigma_u, f.scale), abs=1e-6)
    for ku, ke in ((1.1, 1.0), (0.9, 1.0), (1.0, 1.05), (1.0, 0.95)):
        assert dense(ku * f.sigma_u, ke * f.scale) < f.loglik
    assert 0.6 < f.sigma_u < 1.5 and 0.4 < f.scale < 0.6        # simulated with about 1.0 and 0.5


def test_glmms_recover_the_parameters_of_simulated_data():
    rng = np.random.default_rng(5)
    G, n = 40, 25
    units = np.repeat(np.arange(G), n)
    x = np.repeat(rng.choice([-1.0, 0.0, 1.0], G), n)
    X = np.c_[np.ones(G * n), x]
    u = np.repeat(rng.normal(0, 0.4, G), n)
    E = rng.integers(50, 150, G * n).astype(float)
    y = rng.poisson(rng.gamma(3.0, E * np.exp(-3.0 - 0.6 * x + u) / 3.0)).astype(float)
    f = NB2GLMM(y, X, ["Intercept", "x"], units, offset=np.log(E)).fit()
    assert f.converged and abs(f.beta[1] + 0.6) < 3 * f.se[1] and 0.2 < f.sigma_u < 0.6 and 1.5 < f.scale < 6
    tr = rng.integers(2, 30, G * n).astype(float)
    pr = 1 / (1 + np.exp(-(0.2 - 0.8 * x + u)))
    yb = rng.binomial(tr.astype(int), rng.beta(pr * 10, (1 - pr) * 10)).astype(float)
    fb = BetaBinGLMM(yb, X, ["Intercept", "x"], units, trials=tr).fit()
    assert fb.converged and abs(fb.beta[1] + 0.8) < 3 * fb.se[1] and 4 < fb.scale < 25


def test_signed_root_gives_a_one_sided_p_value():
    rows = [r for r in episodes(seed=2) if r["mode"] == "B"]
    m = P.model_veff(rows)
    r = signed_root(m.fit(), m.drop("rho").fit(), "rho")
    assert r < -3.0                                              # the statistic carries the sign
    down = bootstrap_signed_root(m, "rho", -1, 99, np.random.default_rng(1))
    up = bootstrap_signed_root(m, "rho", +1, 99, np.random.default_rng(1))
    assert down["converged"] and down["p_boot"] == pytest.approx(0.01) and down["p_wald"] < 1e-3
    assert up["p_boot"] == pytest.approx(1.0) and up["p_wald"] > 0.999


def test_a_model_that_does_not_converge_is_an_explicit_failure():
    rows = episodes(seed=3)
    for r in rows:
        r["N_exec"] = 0                                          # no executed lane change at all
    res = P.h2_nexec_rho(rows, 5, np.random.default_rng(0))
    assert res["converged"] is False and res["p_boot"] is None and res["message"]
    out = P.run_primary(rows, 9, 19, seed=1)
    bad = [r for r in out if r["function"] in ("h2_nexec_rho", "h2_nexec_D")]
    assert len(out) == 7 and all(r["outcome"] == "not established (model failure)" for r in bad)
    assert all(r["p_holm_primary"] == 1.0 for r in bad)          # p = 1 inside the family of seven


# ---------------------------------------------------------------------- shared scenario ids
def test_sensitivity_resampling_keeps_shared_scenario_ids():
    c = P.columns(episodes(seed=4))
    draw = P.two_way_resampler(c)
    wu, ws = draw(np.random.default_rng(0))
    for d in (10.0, 25.0, 40.0):
        for s in np.unique(c["episode_seed"]):
            sel = (c["D"] == d) & (c["episode_seed"] == s)
            assert len(set(ws[sel])) == 1                        # one multiplicity for every unit
            assert sel.sum() == 3 * 3 * 2                        # 3 units x 3 rho x 2 modes share the scenario
        assert ws[(c["D"] == d) & (c["unit"] == f"A:{int(d)}:0") & (c["rho"] == 0.4)].sum() == 6
    for u in np.unique(c["unit"]):
        assert len(set(wu[c["unit"] == u])) == 1                 # a unit is drawn as a whole
    a_units = {u: wu[c["unit"] == u][0] for u in np.unique(c["unit"]) if u.startswith("A:10:")}
    assert sum(a_units.values()) == 3                            # resampled inside the cell


def test_plan_runs_end_to_end_and_reports_every_check(tmp_path, capsys):
    rows = episodes(seed=6)
    for i, r in enumerate(r for r in rows if r["mode"] == "A" and r["D"] == 10):
        if i < 6:
            r.update(mode="B2", rho="", unit=f"B2:10:{i % 2}", crashed=1, h1_event=int(i % 3 == 0),
                     crash_category="CR_ego_op (reaction)" if i % 3 == 0 else "CR_env_op (rear-end)", rho_cls=1.0)
    path = tmp_path / "episodes.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert R.main(["--episodes", str(path), "--out", str(tmp_path), "--boot", "39", "--boot-sens", "99",
                   "--boot-diff", "200"]) == 0
    text = capsys.readouterr().out
    for c in P.PRIMARY:
        assert f"`{c.function.__name__}`" in text and c.key in text
    for needle in ("H1a (IUT", "H1b (IUT", "Descriptive difference", "control_natt_mode", "Contrast of differences",
                   "Page's test", "Pearson dispersion", "p Holm", "sens. p Holm"):
        assert needle in text, needle
    out = P.run_primary([r for r in rows if r["mode"] in "AB"], 39, 99, seed=20261110)
    by = {r["function"]: r for r in out}
    for name in ("h2_veff_rho", "h2_veff_D", "h3b_pblock_mode"):
        assert by[name]["primary"]["estimate"] < 0 and by[name]["sensitivity"]["estimate"] < 0
        assert by[name]["primary"]["converged"]
    allowed = {P.ESTABLISHED} | {f"not established ({w})" for w in (
        "model failure", "sensitivity analysis failed", "direction", "primary analysis", "sensitivity analysis")}
    assert {r["outcome"] for r in out} <= allowed
    # with enough replicates the smallest attainable p-value passes Holm for seven checks
    strong = P.h2_veff_rho(rows, 199, np.random.default_rng(1))
    assert strong["p_boot"] == pytest.approx(1 / 200) and 7 * strong["p_boot"] <= P.ALPHA


def test_page_test_and_contrasts():
    L, p, how = P.page_test([[3, 2, 1], [9, 5, 4], [7, 6, 2]])           # decreasing in every block
    assert how == "exact" and p == pytest.approx((1 / 6) ** 3) and L == 3 * (1 * 1 + 2 * 2 + 3 * 3)
    assert P.page_test([[1, 2, 3], [4, 5, 9]])[1] == pytest.approx(1.0)
    means = {(d, r): 20 - 0.1 * d - 5 * r - 0.1 * d * r for d in (10.0, 25.0, 40.0) for r in (0.4, 0.6, 1.0)}
    assert P.contrast_of_differences(means) == pytest.approx(-0.1 * 30 * 0.6)
    adj = P.adjacent_contrasts(means)
    assert set(adj) == {"rho 0.4 -> 0.6", "rho 0.6 -> 1", "D 10 -> 25", "D 25 -> 40"}
    assert all(v < 0 for v in adj.values())
