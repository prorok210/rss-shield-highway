"""Completeness, policy identity, shared parsers and per-hypothesis row selection (protocol 7.9).

The IUT and the mixed models are only defined on the full design: every density, every rho level,
both modes and five trained units per cell, with the fixed scenario sets, consistent cross-cell
policy identity (Mode A preserves unit across rho within D; Mode B has a unique unit per
(D, rho, seed)), and without duplicate episodes.  This check runs before any IUT or model fit in
the main mode; --design pilot marks the output as diagnostic and skips completeness enforcement.

Input table validation is strictly separated from per-hypothesis row selection:
  * descriptive rho levels (0.1, 0.2 s) are allowed in the input table and kept in descriptive
    summaries (including H1 descriptive rows), while remaining strictly excluded from the H1a IUT
    (0.6, 1.0 s) and from primary H2 / H3 / control M5 / H2 secondary monotonicity analyses
    (0.4, 0.6, 1.0 s);
  * boolean flags (correct_start, crashed, h1_event) are parsed through one strict parser
    (`parse_bool`) shared by design validation and H1 calculations.
"""
from dataclasses import dataclass
import math

import numpy as np


class DesignCompletenessError(ValueError):
    """The episode table does not satisfy the pre-registered design contract."""


@dataclass(frozen=True)
class DesignSpec:
    densities: tuple[int, ...] = (10, 25, 40)
    h1a_rhos: tuple[float, ...] = (0.6, 1.0)
    h23_rhos: tuple[float, ...] = (0.4, 0.6, 1.0)
    descriptive_rhos: tuple[float, ...] = (0.1, 0.2)
    h1a_arms: tuple[str, ...] = ("A", "B")
    b2_arm: str = "B2"
    n_units: int = 5
    common_scenarios: frozenset[int] = frozenset(range(100_000, 101_000))
    b2_scenarios: frozenset[int] = frozenset(range(100_000, 101_500))


def normalize_mode(value):
    if value is None:
        return ""
    return str(value).strip()


def parse_rho(value):
    """Parse a rho cell into a canonical float rounded to 6 decimals, or None for empty/null."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise DesignCompletenessError(f"cannot parse rho={value!r}")
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError) as exc:
        raise DesignCompletenessError(f"cannot parse rho={value!r}") from exc
    if not math.isfinite(fv):
        raise DesignCompletenessError(f"cannot parse rho={value!r}")
    return round(fv, 6)


_rho_key = parse_rho


def parse_bool(value, field="correct_start"):
    """Strict boolean parser shared by validation and H1 calculations.

    Matches rss_shield.accounting._parse_bool semantics: accepts bool / np.bool_, exact 0/1
    integers or finite floats, and case-insensitive strings '1'/'true'/'yes' vs '0'/'false'/'no'.
    Any other value raises DesignCompletenessError (a ValueError subclass).
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        iv = int(value)
        if iv in (0, 1):
            return bool(iv)
        raise DesignCompletenessError(f"cannot parse {field}={value!r}")
    if isinstance(value, (float, np.floating)):
        fv = float(value)
        if math.isfinite(fv) and fv in (0.0, 1.0):
            return bool(int(fv))
        raise DesignCompletenessError(f"cannot parse {field}={value!r}")
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes"):
            return True
        if s in ("0", "false", "no"):
            return False
        raise DesignCompletenessError(f"cannot parse {field}={value!r}")
    raise DesignCompletenessError(f"cannot parse {field}={value!r}")


_as_bool = parse_bool


def _parse_int(value, field):
    if isinstance(value, (bool, np.bool_)):
        raise DesignCompletenessError(f"cannot parse {field}={value!r}")
    try:
        fv = float(value)
    except (TypeError, ValueError) as exc:
        raise DesignCompletenessError(f"cannot parse {field}={value!r}") from exc
    if not math.isfinite(fv) or int(fv) != fv:
        raise DesignCompletenessError(f"cannot parse {field}={value!r}")
    return int(fv)


def _rho_set(rhos):
    return frozenset(round(float(r), 6) for r in rhos)


def _allowed_rhos(spec):
    return _rho_set((*spec.h1a_rhos, *spec.h23_rhos, *spec.descriptive_rhos))


def _allowed_modes(spec):
    return frozenset((*spec.h1a_arms, spec.b2_arm))


# ---------------------------------------------------------------------- per-hypothesis selectors
def select_h1a_rows(rows, spec=DesignSpec()):
    """Rows entering the H1a IUT decision: filtered arms (A, B) at rho in spec.h1a_rhos."""
    arms = set(spec.h1a_arms)
    rhos = _rho_set(spec.h1a_rhos)
    return [r for r in rows
            if normalize_mode(r.get("mode")) in arms and parse_rho(r.get("rho")) in rhos]


def select_h1b_rows(rows, spec=DesignSpec()):
    """Rows entering the H1b IUT decision: Baseline 2 (spec.b2_arm)."""
    return [r for r in rows if normalize_mode(r.get("mode")) == spec.b2_arm]


def select_h1_diff_rows(rows, spec=DesignSpec()):
    """Rows entering the paired descriptive difference CR_ego_op(Proposed B, rho) - CR_ego_op(B2)
    for rho in spec.h1a_rhos."""
    rhos = _rho_set(spec.h1a_rhos)
    out = []
    for r in rows:
        m = normalize_mode(r.get("mode"))
        if m == spec.b2_arm or (m == "B" and parse_rho(r.get("rho")) in rhos):
            out.append(r)
    return out


def select_h2_rows(rows, spec=DesignSpec(), *, rhos=None):
    """Rows entering H2 primary models (M1 v_eff, M2 N_exec) and H2 secondary contrasts / Page's
    test: Mode B at rho in spec.h23_rhos ({0.4, 0.6, 1.0} s; excludes descriptive 0.1 and 0.2 s)."""
    allowed = _rho_set(spec.h23_rhos if rhos is None else rhos)
    return [r for r in rows
            if normalize_mode(r.get("mode")) == "B" and parse_rho(r.get("rho")) in allowed]


def select_h3_rows(rows, spec=DesignSpec(), *, rhos=None):
    """Rows entering H3 primary models (M3 N_blocked, M4 P(block|attempt)) and control M5 (N_att):
    Modes A and B at rho in spec.h23_rhos ({0.4, 0.6, 1.0} s; excludes descriptive 0.1 and 0.2 s)."""
    arms = set(spec.h1a_arms)
    allowed = _rho_set(spec.h23_rhos if rhos is None else rhos)
    return [r for r in rows
            if normalize_mode(r.get("mode")) in arms and parse_rho(r.get("rho")) in allowed]


def select_descriptive_rows(rows, spec=DesignSpec()):
    """Rows entering descriptive reporting (all valid arms and rho levels, including 0.1 and 0.2 s)."""
    arms = set(spec.h1a_arms)
    all_rhos = _allowed_rhos(spec)
    out = []
    for r in rows:
        m = normalize_mode(r.get("mode"))
        if m == spec.b2_arm or (m in arms and parse_rho(r.get("rho")) in all_rhos):
            out.append(r)
    return out


# ---------------------------------------------------------------------- policy identity & completeness
def validate_policy_identity(rows, spec=DesignSpec()):
    """Validate cross-cell identity of trained units across modes, densities, rho and unit_seed.

    Rules (protocol 7.1, 7.9):
      * unit must be non-empty and pairwise disjoint across modes (A, B, B2);
      * Mode A: each frozen Baseline 1 policy keeps the exact same unit across all rho within D;
        any descriptive rho cell for (A, D) may only use units from that (A, D) policy set; Mode A
        units are disjoint across different densities D;
      * Mode B: every unit uniquely denotes one training run and appears in strictly one (D, rho)
        cell across all present Mode B cells (primary and descriptive);
      * Mode B2: every unit appears in strictly one density D;
      * when unit_seed is present, it must be present and non-empty on every row, and form a strict
        1-to-1 bijection with unit: (D, unit_seed) <-> unit for Mode A and B2, and
        (D, rho, unit_seed) <-> unit for Mode B.
    """
    if not rows:
        raise DesignCompletenessError("empty episode table")

    allowed_modes = _allowed_modes(spec)
    allowed_rhos = _allowed_rhos(spec)
    h23_rhos = _rho_set(spec.h23_rhos)
    desc_rhos = _rho_set(spec.descriptive_rhos)
    allowed_densities = set(spec.densities)

    has_unit_seed = any("unit_seed" in r for r in rows)
    if has_unit_seed and not all("unit_seed" in r for r in rows):
        raise DesignCompletenessError("unit_seed column is missing on some rows")

    unit_modes = {}
    mode_a_by_D_rho = {}
    mode_a_unit_Ds = {}
    mode_b_by_D_rho = {}
    mode_b_unit_cells = {}
    b2_by_D = {}
    b2_unit_Ds = {}
    key_to_unit = {}
    unit_to_key = {}

    for r in rows:
        mode = normalize_mode(r.get("mode"))
        if mode not in allowed_modes:
            raise DesignCompletenessError(f"unexpected mode outside the design: {mode!r}")
        D = _parse_int(r.get("D"), "D")
        if D not in allowed_densities:
            raise DesignCompletenessError(f"unexpected density D outside the design: {D}")
        unit = "" if r.get("unit") is None else str(r.get("unit")).strip()
        if not unit:
            raise DesignCompletenessError(f"empty unit identifier for mode={mode}, D={D}")

        rho = None if mode == spec.b2_arm else parse_rho(r.get("rho"))
        if mode != spec.b2_arm and (rho is None or rho not in allowed_rhos):
            raise DesignCompletenessError(
                f"unexpected rho={r.get('rho')!r} for mode={mode}, D={D}")

        unit_modes.setdefault(unit, set()).add(mode)
        if mode == "A":
            mode_a_by_D_rho.setdefault((D, rho), set()).add(unit)
            mode_a_unit_Ds.setdefault(unit, set()).add(D)
        elif mode == "B":
            mode_b_by_D_rho.setdefault((D, rho), set()).add(unit)
            mode_b_unit_cells.setdefault(unit, set()).add((D, rho))
        elif mode == spec.b2_arm:
            b2_by_D.setdefault(D, set()).add(unit)
            b2_unit_Ds.setdefault(unit, set()).add(D)

        if has_unit_seed:
            raw_us = r.get("unit_seed")
            us = "" if raw_us is None else str(raw_us).strip()
            if not us:
                raise DesignCompletenessError(
                    f"empty unit_seed for mode={mode}, D={D}, rho={rho}, unit={unit!r}")
            run_key = (mode, D, rho, us) if mode == "B" else (mode, D, us)
            prev_u = key_to_unit.get(run_key)
            if prev_u is not None and prev_u != unit:
                raise DesignCompletenessError(
                    f"inconsistent unit for {run_key}: {prev_u!r} vs {unit!r}")
            key_to_unit[run_key] = unit
            prev_k = unit_to_key.get(unit)
            if prev_k is not None and prev_k != run_key:
                raise DesignCompletenessError(
                    f"unit {unit!r} maps to multiple training runs: {prev_k} vs {run_key}")
            unit_to_key[unit] = run_key

    for unit, modes in sorted(unit_modes.items()):
        if len(modes) > 1:
            raise DesignCompletenessError(
                f"unit {unit!r} is shared across multiple modes: {sorted(modes)}")

    for unit, ds in sorted(mode_a_unit_Ds.items()):
        if len(ds) > 1:
            raise DesignCompletenessError(
                f"Mode A unit {unit!r} is shared across densities {sorted(ds)}")

    mode_a_densities = sorted({d for d, _ in mode_a_by_D_rho})
    for D in mode_a_densities:
        prim_rhos = sorted(r for d, r in mode_a_by_D_rho if d == D and r in h23_rhos)
        desc_present = sorted(r for d, r in mode_a_by_D_rho if d == D and r in desc_rhos)
        if prim_rhos:
            ref_rho = prim_rhos[0]
            ref_units = mode_a_by_D_rho[(D, ref_rho)]
            if spec.n_units is not None and len(ref_units) != spec.n_units:
                raise DesignCompletenessError(
                    f"Mode A (D={D}, rho={ref_rho}): {len(ref_units)} unit(s), expected {spec.n_units}")
            for r in prim_rhos[1:]:
                got = mode_a_by_D_rho[(D, r)]
                if got != ref_units:
                    raise DesignCompletenessError(
                        f"Mode A policy units for D={D} changed across rho: "
                        f"rho={ref_rho} has {sorted(ref_units)}, rho={r} has {sorted(got)}")
            for r in desc_present:
                extra = mode_a_by_D_rho[(D, r)] - ref_units
                if extra:
                    raise DesignCompletenessError(
                        f"Mode A descriptive cell (D={D}, rho={r}) has unit(s) {sorted(extra)} "
                        f"outside the frozen Mode A policy set {sorted(ref_units)}")
        elif desc_present:
            ref_rho = desc_present[0]
            ref_units = mode_a_by_D_rho[(D, ref_rho)]
            for r in desc_present[1:]:
                if mode_a_by_D_rho[(D, r)] != ref_units:
                    raise DesignCompletenessError(
                        f"Mode A descriptive policy units for D={D} changed across rho: "
                        f"rho={ref_rho} vs rho={r}")

    for unit, cells in sorted(mode_b_unit_cells.items()):
        if len(cells) > 1:
            raise DesignCompletenessError(
                f"Mode B unit {unit!r} is reused across multiple (D, rho) cells: {sorted(cells)}")

    if spec.n_units is not None:
        for (D, rho), units in sorted(mode_b_by_D_rho.items()):
            if rho in h23_rhos and len(units) != spec.n_units:
                raise DesignCompletenessError(
                    f"Mode B (D={D}, rho={rho}): {len(units)} unit(s), expected {spec.n_units}")

    for unit, ds in sorted(b2_unit_Ds.items()):
        if len(ds) > 1:
            raise DesignCompletenessError(
                f"Mode {spec.b2_arm} unit {unit!r} is shared across densities {sorted(ds)}")

    if spec.n_units is not None:
        for D, units in sorted(b2_by_D.items()):
            if len(units) != spec.n_units:
                raise DesignCompletenessError(
                    f"Mode {spec.b2_arm} (D={D}): {len(units)} unit(s), expected {spec.n_units}")

    return None


def _check_group(by_cell, label, expected, scenarios, n_units):
    present = {(c[0], c[1], c[2]) for c in by_cell if (c[0], c[1], c[2]) in expected}
    missing = sorted(expected - present)
    if missing:
        raise DesignCompletenessError(
            f"missing {len(missing)} {label} cell(s), e.g. {missing[:3]}")
    for key in sorted(expected):
        cell_units = {c[3] for c in by_cell if (c[0], c[1], c[2]) == key}
        if len(cell_units) != n_units:
            raise DesignCompletenessError(
                f"{label} {key}: {len(cell_units)} unit(s), expected {n_units}")
        for unit in sorted(cell_units):
            if by_cell[(key[0], key[1], key[2], unit)] != set(scenarios):
                raise DesignCompletenessError(
                    f"{label} {(key[0], key[1], key[2], unit)}: scenario set differs from "
                    f"the fixed set of {len(scenarios)}")


def validate_design_completeness(rows, spec=DesignSpec()):
    """Raise DesignCompletenessError unless every pre-registered cell, scenario set, boolean flag,
    and cross-cell policy identity invariant holds.

    Descriptive rho levels (spec.descriptive_rhos, default (0.1, 0.2)) are optional in the input
    table; when present they are checked for duplicates, strict boolean parsing, correct_start
    consistency and Mode A/B policy identity, and are excluded from H1a / H2 / H3 by the explicit
    per-hypothesis selectors.  The rho of Baseline 2 is ignored (classified once at rho_ref = 1.0 s).
    """
    if not rows:
        raise DesignCompletenessError("empty episode table")

    allowed_modes = _allowed_modes(spec)
    allowed_densities = set(spec.densities)
    allowed_rhos = _allowed_rhos(spec)

    by_cell = {}
    valid_start = {}

    for r in rows:
        mode = normalize_mode(r.get("mode"))
        if mode not in allowed_modes:
            raise DesignCompletenessError(
                f"unexpected mode outside the design: {mode!r}")
        D = _parse_int(r.get("D"), "D")
        if D not in allowed_densities:
            raise DesignCompletenessError(
                f"unexpected density D outside the design: {[D]}")
        unit = "" if r.get("unit") is None else str(r.get("unit")).strip()
        if not unit:
            raise DesignCompletenessError(
                f"empty unit identifier for mode={mode}, D={D}")
        seed = _parse_int(r.get("episode_seed"), "episode_seed")
        rho = None if mode == spec.b2_arm else parse_rho(r.get("rho"))
        if mode != spec.b2_arm and (rho is None or rho not in allowed_rhos):
            raise DesignCompletenessError(
                f"unexpected rho={r.get('rho')!r} outside the design for mode={mode}")

        cell = (mode, D, rho, unit)
        episodes = by_cell.setdefault(cell, set())
        if seed in episodes:
            raise DesignCompletenessError(
                f"duplicate row for cell={cell}, episode_seed={seed}")
        episodes.add(seed)

        if "correct_start" not in r:
            raise DesignCompletenessError(
                f"missing correct_start for cell={cell}, episode_seed={seed}")
        vs = parse_bool(r["correct_start"], "correct_start")
        key = (D, seed)
        if key in valid_start and valid_start[key] != vs:
            raise DesignCompletenessError(
                f"inconsistent correct_start for (D={D}, episode_seed={seed}) across cells")
        valid_start[key] = vs

        if "h1_event" in r:
            parse_bool(r["h1_event"], "h1_event")
        if "crashed" in r:
            parse_bool(r["crashed"], "crashed")

    common = {(m, D, round(float(rho), 6)) for m in spec.h1a_arms for D in spec.densities
              for rho in spec.h1a_rhos}
    _check_group(by_cell, "H1a", common, spec.common_scenarios, spec.n_units)

    h1b = {(spec.b2_arm, D, None) for D in spec.densities}
    _check_group(by_cell, "H1b", h1b, spec.b2_scenarios, spec.n_units)

    h23 = {(m, D, round(float(rho), 6)) for m in spec.h1a_arms for D in spec.densities
           for rho in spec.h23_rhos}
    _check_group(by_cell, "H2/H3", h23, spec.common_scenarios, spec.n_units)

    validate_policy_identity(rows, spec)
    return None
