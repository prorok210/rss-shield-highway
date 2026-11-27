"""Regression tests for remarks 4-7 and instructor follow-up remarks 1-2:
design completeness, cross-cell policy identity (Mode A vs Mode B), per-hypothesis row selection
(rho in {0.1, 0.2} descriptive-only, excluded from H1 IUT / H2 / H3 / M5), unified strict boolean
parsing for correct_start, paired H1 bootstrap, Page ties, and the LMM/GLMM convergence contract
with the B_conv + 1 denominator."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import binomial as B
from analysis import plan as P
from analysis import run as R
from analysis.design import (DesignCompletenessError, DesignSpec, parse_bool,
                             select_descriptive_rows, select_h1_diff_rows, select_h1a_rows,
                             select_h1b_rows, select_h2_rows, select_h3_rows,
                             validate_design_completeness, validate_policy_identity)
from analysis.models import (bootstrap_lr_pvalue, check_glmm_convergence, check_lmm_convergence)

REPO = Path(__file__).resolve().parents[2]

SMALL = DesignSpec(densities=(10,), h1a_rhos=(0.6,), h23_rhos=(0.4, 0.6), n_units=2,
                   common_scenarios=frozenset({1, 2, 3}), b2_scenarios=frozenset({1, 2, 3, 4}))

MULTI = DesignSpec(densities=(10, 25), h1a_rhos=(0.6, 1.0), h23_rhos=(0.4, 0.6, 1.0),
                   descriptive_rhos=(0.1, 0.2), n_units=2,
                   common_scenarios=frozenset({1, 2, 3}), b2_scenarios=frozenset({1, 2, 3, 4}))


def _design_rows(spec=SMALL, *, include_unit_seed=True, extra_rhos=(), correct_start=1):
    """Canonical design table: Mode A keeps unit across rho within D; Mode B has a unique unit per
    (D, rho, unit_seed); Mode B2 has a unique unit per (D, unit_seed)."""
    rows = []
    rhos = list(spec.h23_rhos) + list(extra_rhos)
    for D in spec.densities:
        for mode in spec.h1a_arms:
            for rho in rhos:
                for u in range(spec.n_units):
                    unit = f"A:{D}:{u}" if mode == "A" else f"B:{D}:{rho}:{u}"
                    for s in sorted(spec.common_scenarios):
                        r = dict(mode=mode, D=D, rho=rho, unit=unit, episode_seed=s,
                                 correct_start=correct_start)
                        if include_unit_seed:
                            r["unit_seed"] = u
                        rows.append(r)
        for u in range(spec.n_units):
            for s in sorted(spec.b2_scenarios):
                r = dict(mode=spec.b2_arm, D=D, rho="", unit=f"{spec.b2_arm}:{D}:{u}",
                         episode_seed=s, correct_start=correct_start)
                if include_unit_seed:
                    r["unit_seed"] = u
                rows.append(r)
    return rows


def _full_episode_rows(spec, *, seed=0, extra_rhos=(), sentinel_extra=False, correct_start=1):
    """Synthetic per-episode table with all columns required by analysis.plan and analysis.run."""
    rng = np.random.default_rng(seed)
    rows = []
    rhos = list(spec.h23_rhos) + list(extra_rhos)
    desc_set = set(spec.descriptive_rhos)
    for D in spec.densities:
        for u in range(spec.n_units):
            ua = float(rng.normal(0, 0.15))
            for rho in rhos:
                is_sentinel = sentinel_extra and (rho in desc_set)
                ub = float(rng.normal(0, 0.15))
                for mode, eff_u, unit in (("A", ua, f"A:{D}:{u}"),
                                          ("B", ub, f"B:{D}:{rho}:{u}")):
                    for s in sorted(spec.common_scenarios):
                        dc, rc, m = float(P.Dc(D)), float(P.rhoc(rho)), float(mode == "B")
                        if is_sentinel:
                            E, v_eff, n_exec, att, blk = 150, 1e6, 500, 600, 500
                            crashed, h1_ev = 1, 1
                            cat = "CR_ego_op (reaction)"
                        else:
                            E = int(rng.integers(120, 200))
                            att = int(max(2, rng.poisson(E * np.exp(-2.0 + eff_u))))
                            pb = 1 / (1 + np.exp(-(0.3 - 1.2 * m + 0.4 * dc + 0.5 * rc + eff_u)))
                            blk = int(rng.binomial(att, pb))
                            v_eff = float(18 - (2.0 * dc + 6.0 * rc + 3.0 * dc * rc)
                                          + eff_u + rng.normal(0, 0.5))
                            n_exec = int(rng.poisson(E * np.exp(-3.0 - (0.5 * dc + 1.5 * rc) + eff_u)))
                            crashed, h1_ev, cat = 0, 0, ""
                        rows.append(dict(
                            mode=mode, D=D, rho=rho, unit=unit, unit_seed=u, episode_seed=s,
                            correct_start=correct_start, rho_cls=rho, crashed=crashed,
                            crash_category=cat, h1_event=h1_ev, E=E, N_att=att, N_blocked=blk,
                            N_exec=n_exec, v_eff=v_eff,
                        ))
        for u in range(spec.n_units):
            for s in sorted(spec.b2_scenarios):
                ev = int(s % 2 == 0)
                rows.append(dict(
                    mode=spec.b2_arm, D=D, rho="", unit=f"{spec.b2_arm}:{D}:{u}",
                    unit_seed=u, episode_seed=s, correct_start=correct_start, rho_cls=1.0,
                    crashed=1, crash_category="CR_ego_op (reaction)" if ev else "CR_env_op (rear-end)",
                    h1_event=ev, E=100, N_att=15, N_blocked=0, N_exec=3, v_eff=10.0,
                ))
    return rows


# ---------------------------------------------------------------- design completeness (existing)
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


# ---------------------------------------------------------------- remark 1: cross-cell policy identity
def test_policy_identity_accepts_valid_multi_density_grid_and_sets_expected_model_groups():
    for with_seed in (True, False):
        rows = _design_rows(MULTI, include_unit_seed=with_seed)
        assert validate_design_completeness(rows, MULTI) is None
        assert validate_policy_identity(rows, MULTI) is None

    full = _full_episode_rows(MULTI, seed=10)
    assert validate_design_completeness(full, MULTI) is None
    # Mode B: 2 densities x 3 primary rhos x 2 units = 12 distinct training runs
    assert P.model_veff(full).G == 12
    # Modes A + B: Mode A has 2 densities x 2 frozen policies = 4 units (preserved across rho)
    # plus 12 Mode B training runs = 16 total random-intercept units
    assert P.model_nblocked(full).G == 16


@pytest.mark.parametrize("include_unit_seed", [True, False])
def test_policy_identity_rejects_mode_a_separate_unit_per_rho(include_unit_seed):
    """Instructor counterexample: giving a frozen Mode A policy a separate unit for each rho must
    fail validation even though every (A, D, rho) cell still has n_units units."""
    rows = _design_rows(MULTI, include_unit_seed=include_unit_seed)
    for r in rows:
        if r["mode"] == "A":
            u_idx = r["unit"].split(":")[-1]
            r["unit"] = f"A:{r['D']}:{r['rho']}:{u_idx}"
    with pytest.raises(DesignCompletenessError, match="Mode A|inconsistent unit"):
        validate_design_completeness(rows, MULTI)
    with pytest.raises(DesignCompletenessError, match="Mode A|inconsistent unit"):
        validate_policy_identity(rows, MULTI)


def test_policy_identity_rejects_mode_a_unit_seed_swap_across_rho():
    """Even if the set of Mode A unit names matches across rho, swapping which unit_seed maps to
    which unit between rho levels must be rejected."""
    rows = _design_rows(MULTI, include_unit_seed=True)
    for r in rows:
        if r["mode"] == "A" and r["D"] == 10 and r["rho"] == 0.6:
            r["unit"] = "A:10:1" if r["unit_seed"] == 0 else "A:10:0"
    with pytest.raises(DesignCompletenessError, match="inconsistent unit"):
        validate_design_completeness(rows, MULTI)


def test_policy_identity_rejects_mode_a_and_b2_unit_shared_across_densities():
    rows_a = _design_rows(MULTI, include_unit_seed=False)
    for r in rows_a:
        if r["mode"] == "A" and r["unit"].endswith(":0"):
            r["unit"] = "A:shared:0"
    with pytest.raises(DesignCompletenessError, match="Mode A unit"):
        validate_design_completeness(rows_a, MULTI)

    rows_b2 = _design_rows(MULTI, include_unit_seed=False)
    for r in rows_b2:
        if r["mode"] == "B2" and r["unit"].endswith(":0"):
            r["unit"] = "B2:shared:0"
    with pytest.raises(DesignCompletenessError, match="B2 unit"):
        validate_design_completeness(rows_b2, MULTI)


@pytest.mark.parametrize("include_unit_seed", [True, False])
def test_policy_identity_rejects_mode_b_unit_shared_across_rho_or_density(include_unit_seed):
    """Mode B must uniquely identify an independent training run by (D, rho, seed)."""
    # Shared across rho within D = 10
    rows_rho = _design_rows(MULTI, include_unit_seed=include_unit_seed)
    for r in rows_rho:
        if r["mode"] == "B" and r["D"] == 10 and r["unit"].endswith(":0"):
            r["unit"] = "B:10:0"
    with pytest.raises(DesignCompletenessError, match="Mode B unit|multiple training runs"):
        validate_design_completeness(rows_rho, MULTI)

    # Shared across D at rho = 0.4
    rows_d = _design_rows(MULTI, include_unit_seed=include_unit_seed)
    for r in rows_d:
        if r["mode"] == "B" and r["rho"] == 0.4 and r["unit"].endswith(":0"):
            r["unit"] = "B:0.4:0"
    with pytest.raises(DesignCompletenessError, match="Mode B unit|multiple training runs"):
        validate_design_completeness(rows_d, MULTI)


def test_policy_identity_rejects_mode_b_unit_seed_inconsistency():
    # Two different unit_seeds sharing one unit inside a single (B, D, rho) cell
    rows_merge = _design_rows(MULTI, include_unit_seed=True)
    for r in rows_merge:
        if r["mode"] == "B" and r["D"] == 10 and r["rho"] == 0.4 and r["unit_seed"] == 1:
            r["unit"] = "B:10:0.4:0"
            r["episode_seed"] = r["episode_seed"] + 100
    with pytest.raises(DesignCompletenessError, match="multiple training runs"):
        validate_policy_identity(rows_merge, MULTI)

    # One (B, D, rho, unit_seed) split across two unit IDs on different episodes
    rows_split = _design_rows(MULTI, include_unit_seed=True)
    hit = next(r for r in rows_split if r["mode"] == "B" and r["D"] == 10 and r["rho"] == 0.4
               and r["unit_seed"] == 0)
    hit["unit"] = "B:10:0.4:0_dup"
    with pytest.raises(DesignCompletenessError, match="inconsistent unit"):
        validate_policy_identity(rows_split, MULTI)


def test_policy_identity_rejects_cross_mode_collision_and_empty_ids():
    # Same unit string used in Mode A and Mode B
    rows_ab = _design_rows(MULTI, include_unit_seed=False)
    for r in rows_ab:
        if r["mode"] == "B" and r["D"] == 10 and r["rho"] == 0.4 and r["unit"] == "B:10:0.4:0":
            r["unit"] = "A:10:0"
    with pytest.raises(DesignCompletenessError, match="multiple modes"):
        validate_design_completeness(rows_ab, MULTI)

    # Same unit string used in Mode B and B2
    rows_bb2 = _design_rows(MULTI, include_unit_seed=False)
    for r in rows_bb2:
        if r["mode"] == "B2" and r["D"] == 10 and r["unit"] == "B2:10:0":
            r["unit"] = "B:10:0.4:0"
    with pytest.raises(DesignCompletenessError, match="multiple modes"):
        validate_design_completeness(rows_bb2, MULTI)

    # Empty unit
    rows_empty = _design_rows(SMALL)
    rows_empty[0]["unit"] = "   "
    with pytest.raises(DesignCompletenessError, match="empty unit"):
        validate_design_completeness(rows_empty, SMALL)

    # Empty unit_seed
    rows_empty_us = _design_rows(SMALL, include_unit_seed=True)
    rows_empty_us[0]["unit_seed"] = ""
    with pytest.raises(DesignCompletenessError, match="empty unit_seed"):
        validate_design_completeness(rows_empty_us, SMALL)

    # Mixed present / missing unit_seed
    rows_mixed_us = _design_rows(SMALL, include_unit_seed=True)
    del rows_mixed_us[0]["unit_seed"]
    with pytest.raises(DesignCompletenessError, match="unit_seed"):
        validate_design_completeness(rows_mixed_us, SMALL)


def test_policy_identity_enforced_on_descriptive_rho_rows():
    # Valid descriptive rho = 0.1, 0.2: Mode A keeps A:{D}:{u}, Mode B uses fresh B:{D}:{rho}:{u}
    valid_desc = _design_rows(MULTI, extra_rhos=(0.1, 0.2))
    assert validate_design_completeness(valid_desc, MULTI) is None

    # Invalid: descriptive rho = 0.2 invents a new Mode A unit
    bad_a = _design_rows(MULTI, include_unit_seed=False)
    bad_a.append(dict(mode="A", D=10, rho=0.2, unit="A:10:0.2:0", episode_seed=1, correct_start=1))
    with pytest.raises(DesignCompletenessError, match="Mode A descriptive"):
        validate_design_completeness(bad_a, MULTI)

    # Invalid: descriptive rho = 0.2 reuses a primary Mode B unit
    bad_b = _design_rows(MULTI, include_unit_seed=False)
    bad_b.append(dict(mode="B", D=10, rho=0.2, unit="B:10:0.4:0", episode_seed=1, correct_start=1))
    with pytest.raises(DesignCompletenessError, match="Mode B unit"):
        validate_design_completeness(bad_b, MULTI)


def test_pilot_episodes_csv_passes_policy_identity_and_strict_bool():
    path = REPO / "runs" / "pilot_run" / "stats" / "episodes.csv"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 168
    assert validate_policy_identity(rows, DesignSpec(n_units=2)) is None
    for r in rows:
        assert isinstance(parse_bool(r["correct_start"], "correct_start"), bool)
        assert isinstance(parse_bool(r["crashed"], "crashed"), bool)
        assert isinstance(parse_bool(r["h1_event"], "h1_event"), bool)


# ---------------------------------------------------------------- remark 2: selectors & rho 0.1/0.2
def test_completeness_accepts_descriptive_rhos_and_rejects_unexpected_rho_or_mode():
    # Descriptive 0.1 and 0.2 accepted
    rows_desc = _design_rows(SMALL, extra_rhos=(0.1, 0.2))
    assert validate_design_completeness(rows_desc, SMALL) is None

    # Out-of-protocol rho rejected
    for bad_rho in (0.3, 0.8, "", None):
        rows = _design_rows(SMALL, include_unit_seed=False)
        rows.append(dict(mode="B", D=10, rho=bad_rho, unit="B:10:bad:0", episode_seed=1,
                         correct_start=1))
        with pytest.raises(DesignCompletenessError, match="unexpected rho"):
            validate_design_completeness(rows, SMALL)

    # Unknown mode rejected
    rows_mode = _design_rows(SMALL, include_unit_seed=False)
    rows_mode.append(dict(mode="A-raw", D=10, rho=0.4, unit="Araw:10:0", episode_seed=1,
                          correct_start=1))
    with pytest.raises(DesignCompletenessError, match="unexpected mode"):
        validate_design_completeness(rows_mode, SMALL)


def test_descriptive_rhos_01_02_stay_in_descriptive_and_never_leak_into_h1_iut_h2_h3():
    """Instructor probe: adding rho = 0.1 and 0.2 s rows with extreme sentinel values passes
    completeness validation, stays in descriptive reporting (h1_checks + descriptive_cells), and
    produces bit-identical primary H1 IUT, H1 difference, M1-M5 models, sensitivity, dispersion,
    contrasts, and Page's test."""
    spec = DesignSpec(densities=(10, 25, 40), h1a_rhos=(0.6, 1.0), h23_rhos=(0.4, 0.6, 1.0),
                      descriptive_rhos=(0.1, 0.2), n_units=2,
                      common_scenarios=frozenset(range(1, 5)),
                      b2_scenarios=frozenset(range(1, 6)))
    base = _full_episode_rows(spec, seed=21, extra_rhos=())
    with_desc = _full_episode_rows(spec, seed=21, extra_rhos=(0.1, 0.2), sentinel_extra=True)
    # Copy primary rows identically from `base` so RNG draws match bit-for-bit on primary cells
    primary_keys = {(r["mode"], r["D"], r["rho"], r["unit"], r["episode_seed"]) for r in base}
    extra_only = [r for r in with_desc
                  if (r["mode"], r["D"], r["rho"], r["unit"], r["episode_seed"]) not in primary_keys]
    assert len(extra_only) > 0
    combined = base + extra_only

    assert validate_design_completeness(base, spec) is None
    assert validate_design_completeness(combined, spec) is None

    # Explicit selectors separate every slice cleanly
    assert len(select_h1a_rows(combined, spec)) == len(select_h1a_rows(base, spec))
    assert len(select_h1b_rows(combined, spec)) == len(select_h1b_rows(base, spec))
    assert len(select_h1_diff_rows(combined, spec)) == len(select_h1_diff_rows(base, spec))
    assert len(select_h2_rows(combined, spec)) == len(select_h2_rows(base, spec))
    assert len(select_h3_rows(combined, spec)) == len(select_h3_rows(base, spec))
    assert len(select_descriptive_rows(combined, spec)) == len(combined)

    # All 5 models M1-M5 have bit-identical design matrices, outcomes, groups, and MLE fits
    for builder in (P.model_veff, P.model_nexec, P.model_nblocked, P.model_pblock, P.model_natt):
        m0, m1 = builder(base), builder(combined)
        assert m0.G == m1.G
        assert np.array_equal(m0.X, m1.X)
        assert np.array_equal(m0.y, m1.y)
        assert np.array_equal(m0.g, m1.g)
        f0, f1 = m0.fit(), m1.fit()
        assert np.allclose(f0.beta, f1.beta, equal_nan=True)
        assert f0.loglik == pytest.approx(f1.loglik)

    # Full primary family (M1-M4 + two-way sensitivity + Holm + reconciliation) bit-identical
    p0 = P.run_primary(base, 5, 9, seed=123)
    p1 = P.run_primary(combined, 5, 9, seed=123)
    assert p0 == p1

    # Dispersion diagnostics bit-identical
    for args in (("N_exec", P.H2_TERMS, "B", None),
                 ("N_blocked", P.H3_TERMS, "AB", None),
                 ("N_att", P.H3_TERMS, "AB", None),
                 ("N_blocked", P.H3_TERMS, "AB", "N_att")):
        assert P.dispersion(base, *args) == pytest.approx(P.dispersion(combined, *args))

    # Secondary H2 contrasts and Page's tests bit-identical (0.1 / 0.2 excluded)
    m_base, reps_base = P.bootstrap_cells(base, "v_eff", 9, np.random.default_rng(7))
    m_comb, reps_comb = P.bootstrap_cells(combined, "v_eff", 9, np.random.default_rng(7))
    assert m_base == m_comb and reps_base == reps_comb
    assert P.adjacent_contrasts(m_base) == P.adjacent_contrasts(m_comb)
    for over in ("rho", "D"):
        lv0, tab0 = P.trend_table(base, "v_eff", over)
        lv1, tab1 = P.trend_table(combined, "v_eff", over)
        assert lv0 == lv1 and np.array_equal(tab0, tab1)

    # H1a IUT and H1 paired difference ignore sentinel crashes at rho = 0.1 and 0.2
    checks_base = R.h1_checks(base)
    checks_comb = R.h1_checks(combined)
    assert R.h1_summary(checks_base) == R.h1_summary(checks_comb)
    assert R.h1_difference(base, 50, np.random.default_rng(9)) == R.h1_difference(
        combined, 50, np.random.default_rng(9))

    # Descriptive reporting DOES retain rho = 0.1 and 0.2 with their sentinel metrics
    h1_desc = [c for c in checks_comb if c["rho"] in ("0.1", "0.2")]
    assert len(h1_desc) == 3 * 2 * 2 * 2   # 3 D x 2 modes x 2 descriptive rhos x 2 units
    assert all(c["CR_total"] == 1.0 and c["events"] == 4 for c in h1_desc)

    desc = R.descriptive_cells(combined, spec)
    desc_01_02 = [d for d in desc if d["rho"] in ("0.1", "0.2")]
    assert len(desc_01_02) == 3 * 2 * 2  # 3 D x 2 modes x 2 descriptive rhos
    assert all(d["role"] == "descriptive (rho < tau_eff)" for d in desc_01_02)
    assert all(d["v_eff"] == pytest.approx(1e6) and d["CR_total"] == 1.0 and d["CR_ego_op"] == 1.0
               for d in desc_01_02)


def test_partial_descriptive_rho_02_runs_end_to_end_without_leaking_or_keyerror(tmp_path):
    """Appending rho = 0.2 rows for only one density (Mode B, D = 10) must not crash
    adjacent_contrasts with KeyError and must leave M1-M5 estimates bit-identical."""
    spec = DesignSpec(densities=(10, 25, 40), h1a_rhos=(0.6, 1.0), h23_rhos=(0.4, 0.6, 1.0),
                      descriptive_rhos=(0.1, 0.2), n_units=2,
                      common_scenarios=frozenset(range(1, 5)),
                      b2_scenarios=frozenset(range(1, 6)))
    base = _full_episode_rows(spec, seed=31)
    partial = list(base)
    for s in sorted(spec.common_scenarios):
        partial.append(dict(
            mode="B", D=10, rho=0.2, unit="B:10:0.2:0", unit_seed=0, episode_seed=s,
            correct_start=1, rho_cls=0.2, crashed=1, crash_category="CR_ego_op (reaction)",
            h1_event=1, E=100, N_att=50, N_blocked=45, N_exec=25, v_eff=9999.0,
        ))
    assert validate_design_completeness(partial, spec) is None

    def run_cli(rows, subdir):
        d = tmp_path / subdir
        d.mkdir()
        p = d / "episodes.csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        rc = R.main(["--episodes", str(p), "--out", str(d), "--boot", "9", "--boot-sens", "19",
                     "--boot-diff", "50", "--design", "pilot", "--seed", "77"])
        assert rc == 0
        return json.loads((d / "analysis.json").read_text(encoding="utf-8"))

    j_base = run_cli(base, "base")
    j_part = run_cli(partial, "partial")
    assert j_base["primary"] == j_part["primary"]
    assert j_base["control"] == j_part["control"]
    assert j_base["dispersion"] == j_part["dispersion"]
    assert j_base["contrast"] == j_part["contrast"]
    assert j_base["adjacent"] == j_part["adjacent"]
    assert j_base["page"] == j_part["page"]
    assert j_base["h1"] == j_part["h1"]
    assert j_base["h1_difference"] == j_part["h1_difference"]
    assert j_base["excluded"] == j_part["excluded"]
    assert any(d["rho"] == "0.2" and d["v_eff"] == pytest.approx(9999.0)
               for d in j_part["descriptive"])
    assert any(c["rho"] == "0.2" and c["events"] == 4 for c in j_part["h1_checks"])


# ---------------------------------------------------------------- remark 2: unified correct_start
@pytest.mark.parametrize("raw, expected", [
    (True, True), (False, False), (np.bool_(True), True), (np.bool_(False), False),
    (1, True), (0, False), (np.int64(1), True), (np.int64(0), False),
    (1.0, True), (0.0, False), (np.float64(1.0), True), (np.float64(0.0), False),
    ("1", True), ("0", False), (" 1 ", True), (" 0 ", False),
    ("true", True), ("True", True), ("TRUE", True), (" false ", False), ("FALSE", False),
    ("yes", True), ("YES", True), (" Yes ", True), ("no", False), ("NO", False), (" No ", False),
])
def test_parse_bool_accepts_canonical_values(raw, expected):
    assert parse_bool(raw, "correct_start") is expected


@pytest.mark.parametrize("bad", [
    "", "   ", "maybe", "2", "-1", "1.0", 2, -1, 0.5, None, float("nan"), float("inf"),
])
def test_parse_bool_rejects_invalid_values(bad):
    with pytest.raises(DesignCompletenessError):
        parse_bool(bad, "correct_start")


def test_correct_start_yes_is_consistent_across_validator_and_h1():
    """Instructor counterexample: validator accepted 'yes' while H1 silently treated 'yes' as False.
    With the shared strict parser, 'yes' / 'YES' / ' 1 ' match '1' across validator, h1_checks,
    h1_difference, and descriptive_cells; invalid values raise DesignCompletenessError everywhere."""
    ref_rows = _full_episode_rows(SMALL, seed=42, correct_start="1")
    assert validate_design_completeness(ref_rows, SMALL) is None
    ref_checks = R.h1_checks(ref_rows)
    ref_diff = R.h1_difference(ref_rows, 100, np.random.default_rng(5))
    ref_desc = R.descriptive_cells(ref_rows, SMALL)
    assert all(c["n_valid"] > 0 for c in ref_checks)
    assert len(ref_diff) == 1 and ref_diff[0]["n_common_scenarios"] == 3

    for truthy in ("yes", "YES", " true ", " 1 ", True):
        rows = _full_episode_rows(SMALL, seed=42, correct_start=truthy)
        assert validate_design_completeness(rows, SMALL) is None
        assert R.h1_checks(rows) == ref_checks
        assert R.h1_difference(rows, 100, np.random.default_rng(5)) == ref_diff
        assert R.descriptive_cells(rows, SMALL) == ref_desc

    for falsy in ("no", "NO", "false", "0", False):
        rows = _full_episode_rows(SMALL, seed=42, correct_start=falsy)
        assert validate_design_completeness(rows, SMALL) is None
        assert all(c["n_valid"] == 0 for c in R.h1_checks(rows))
        assert R.h1_difference(rows, 100, np.random.default_rng(5)) == []

    # Cross-cell consistency: 'yes' and '1' mean the same boolean True; 'yes' vs 'no' is inconsistent
    mixed_ok = _full_episode_rows(SMALL, seed=42, correct_start="1")
    for r in mixed_ok:
        if r["mode"] == "B":
            r["correct_start"] = "yes"
    assert validate_design_completeness(mixed_ok, SMALL) is None
    assert R.h1_checks(mixed_ok) == ref_checks

    mixed_bad = _full_episode_rows(SMALL, seed=42, correct_start="yes")
    next(r for r in mixed_bad if r["mode"] == "B")["correct_start"] = "no"
    with pytest.raises(DesignCompletenessError, match="inconsistent correct_start"):
        validate_design_completeness(mixed_bad, SMALL)

    # Malformed correct_start fails loudly in validator AND in H1 (never silently coerced to False)
    malformed = _full_episode_rows(SMALL, seed=42, correct_start="maybe")
    with pytest.raises(DesignCompletenessError, match="correct_start"):
        validate_design_completeness(malformed, SMALL)
    with pytest.raises(DesignCompletenessError, match="correct_start"):
        R.h1_checks(malformed)
    with pytest.raises(DesignCompletenessError, match="correct_start"):
        R.h1_difference(malformed, 10, np.random.default_rng(1))

    # h1_event and crashed also use the shared strict parser
    yes_ev = _full_episode_rows(SMALL, seed=42, correct_start="yes")
    for r in yes_ev:
        if r["mode"] == "B2":
            r["crashed"] = "yes"
            r["h1_event"] = "yes" if r["h1_event"] else "no"
    assert validate_design_completeness(yes_ev, SMALL) is None
    assert R.h1_checks(yes_ev) == ref_checks

    bad_ev = _full_episode_rows(SMALL, seed=42, correct_start="1")
    bad_ev[0]["h1_event"] = "invalid"
    with pytest.raises(DesignCompletenessError, match="h1_event"):
        validate_design_completeness(bad_ev, SMALL)
    with pytest.raises(DesignCompletenessError, match="h1_event"):
        R.h1_checks(bad_ev)


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
