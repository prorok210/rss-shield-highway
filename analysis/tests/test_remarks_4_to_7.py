"""Regression tests for remarks 4-7: design completeness, paired H1 bootstrap, Page ties and
the LMM/GLMM convergence contract with the B_conv + 1 denominator."""
import numpy as np
import pytest

from analysis import binomial as B
from analysis.design import DesignCompletenessError, DesignSpec, validate_design_completeness
from analysis.models import (bootstrap_lr_pvalue, check_glmm_convergence, check_lmm_convergence)


SMALL = DesignSpec(densities=(10,), h1a_rhos=(0.6,), h23_rhos=(0.4, 0.6), n_units=2,
                   common_scenarios=frozenset({1, 2, 3}), b2_scenarios=frozenset({1, 2, 3, 4}))


def _design_rows(spec=SMALL):
    rows = []
    for mode in spec.h1a_arms:
        for rho in spec.h23_rhos:
            for u in range(spec.n_units):
                for s in spec.common_scenarios:
                    rows.append(dict(mode=mode, D=10, rho=rho, unit=f"{mode}:10:{rho}:{u}",
                                     episode_seed=s, correct_start=1))
    for u in range(spec.n_units):
        for s in spec.b2_scenarios:
            rows.append(dict(mode="B2", D=10, rho="", unit=f"B2:10:{u}", episode_seed=s,
                             correct_start=1))
    return rows


def test_design_completeness_accepts_the_full_grid():
    assert validate_design_completeness(_design_rows(), SMALL) is None


def test_design_completeness_rejects_a_missing_seed_and_a_duplicate():
    rows = _design_rows()
    broken = rows[:-1]                                  # one Baseline 2 scenario missing
    with pytest.raises(DesignCompletenessError):
        validate_design_completeness(broken, SMALL)
    with pytest.raises(DesignCompletenessError):
        validate_design_completeness(rows + [rows[0]], SMALL)


def test_design_completeness_rejects_inconsistent_valid_start():
    rows = _design_rows()
    flip = next(r for r in rows if r["mode"] == "B" and r["rho"] == 0.6)
    flip["correct_start"] = 0
    with pytest.raises(DesignCompletenessError):
        validate_design_completeness(rows, SMALL)


def test_design_completeness_rejects_a_missing_cell():
    rows = [r for r in _design_rows() if not (r["mode"] == "A" and r["rho"] == 0.4)]
    with pytest.raises(DesignCompletenessError):
        validate_design_completeness(rows, SMALL)


def test_design_completeness_rejects_unexpected_density():
    rows = _design_rows()
    rows.append(dict(rows[0], D=20, unit="X:20:0", episode_seed=1))
    with pytest.raises(DesignCompletenessError):
        validate_design_completeness(rows, SMALL)


def test_baseline2_rho_is_ignored_by_normalisation():
    rows = _design_rows()
    for r in rows:
        if r["mode"] == "B2":
            r["rho"] = "1.0"
    assert validate_design_completeness(rows, SMALL) is None


# ---------------------------------------------------------------- paired bootstrap
def test_paired_bootstrap_is_exact_on_identical_vectors():
    prop = {0: {1: 1, 2: 0, 3: 1}}
    base = {0: {1: 1, 2: 0, 3: 1, 4: 1}}
    r = B.bootstrap_h1_diff_twoway(prop, base, n_boot=500, seed=7)
    assert r["diff"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["n_common_scenarios"] == 3 and r["n_extra_b2_excluded"] == 1
    assert r["n_extra_prop_excluded"] == 0


def test_paired_bootstrap_exact_for_5x50_identical_vectors():
    # the teacher's counterexample: five trained units per arm, one shared 0/1 vector of 50 events
    vec = {s: int(s % 3 == 0) for s in range(50)}
    prop = {u: dict(vec) for u in range(5)}
    base = {u: dict(vec) for u in range(5)}
    r = B.bootstrap_h1_diff_twoway(prop, base, n_boot=200, seed=11)
    assert r["diff"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["n_common_scenarios"] == 50 and r["n_units_proposed"] == 5


def test_paired_bootstrap_rejects_inconsistent_scenario_sets():
    with pytest.raises(ValueError):
        B.bootstrap_h1_diff_twoway({0: {1: 0}, 1: {2: 0}}, {0: {1: 0, 2: 0}}, n_boot=10)


# ---------------------------------------------------------------- convergence
def _glmm(**kw):
    args = dict(opt_success=True, loglik=-1.0, params=np.array([0.0]), se=np.array([1.0]),
                grad_free=np.array([1e-6]), hess_free=np.array([[1.0]]), quad_diff=0.0)
    args.update(kw)
    return check_glmm_convergence(**args)


def test_lmm_requires_optimizer_success_and_pd_hessian():
    ok = check_lmm_convergence(opt_success=True, loglik=-1.0, params=np.array([0.0]),
                               se=np.array([1.0]), hess_free=np.array([[1.0]]), sigma2=1.0)
    assert ok.converged and ok.converged_via == "optimizer_success"
    assert check_lmm_convergence(opt_success=False, loglik=-1.0, params=np.array([0.0]),
                                 se=np.array([1.0]), hess_free=np.array([[1.0]]),
                                 sigma2=1.0).reason == "optimizer_failed"
    assert not check_lmm_convergence(opt_success=True, loglik=-1.0, params=np.array([0.0]),
                                     se=np.array([1.0]), hess_free=np.array([[1.0, 0.0],
                                                                             [0.0, -1.0]]),
                                     sigma2=1.0).converged


def test_glmm_two_level_convergence_contract():
    assert _glmm().converged_via == "optimizer_success"
    # the optimizer succeeded but the gradient is too large for the first level
    assert not _glmm(grad_free=np.array([5e-3])).converged
    # a failed optimizer is allowed only with the stricter fallback gradient
    fallback = _glmm(opt_success=False, grad_free=np.array([5e-5]))
    assert fallback.converged and fallback.converged_via == "stationary_gradient"
    assert not _glmm(opt_success=False, grad_free=np.array([5e-4])).converged
    # small gradient in a saddle is not convergence
    saddle = _glmm(hess_free=np.array([[-1.0]]), opt_success=False, grad_free=np.array([1e-6]))
    assert not saddle.converged and saddle.reason == "non_pd_hessian"
    assert not _glmm(kkt_boundary_ok=False).converged
    assert not _glmm(quad_diff=0.5).converged
    assert not _glmm(has_events=False).converged


def test_bootstrap_p_uses_B_conv_plus_one_and_fails_above_ten_percent():
    r = bootstrap_lr_pvalue(1.0, [0.0, 2.0] + [3.0] * 7, 10)
    assert r["model_failed"] is False and r["B_conv"] == 9
    assert r["p_value"] == pytest.approx((1 + 1) / (9 + 1))          # not 1 / 10
    failed = bootstrap_lr_pvalue(1.0, [0.0] * 8, 10)
    assert failed["model_failed"] is True and failed["p_value"] == 1.0
    assert failed["fail_frac"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        bootstrap_lr_pvalue(1.0, [], 0)
