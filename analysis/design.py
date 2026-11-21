"""Completeness of the episode table before H1/H2/H3 inference (protocol 7.9).

The IUT and the mixed models are only defined on the full design: every density, every rho level,
both modes and five trained units per cell, with the fixed scenario sets and without duplicate
episodes.  This check runs before any IUT or model fit in the main mode; --design pilot marks the
output as diagnostic and skips it.  A missing cell is a hard error, not an empty result.
"""
from dataclasses import dataclass


class DesignCompletenessError(ValueError):
    """The episode table does not cover the pre-registered design."""


@dataclass(frozen=True)
class DesignSpec:
    densities: tuple[int, ...] = (10, 25, 40)
    h1a_rhos: tuple[float, ...] = (0.6, 1.0)
    h23_rhos: tuple[float, ...] = (0.4, 0.6, 1.0)
    h1a_arms: tuple[str, ...] = ("A", "B")
    b2_arm: str = "B2"
    n_units: int = 5
    common_scenarios: frozenset[int] = frozenset(range(100_000, 101_000))
    b2_scenarios: frozenset[int] = frozenset(range(100_000, 101_500))


def _rho_key(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return round(float(value), 6)


def _as_bool(value):
    s = str(value).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    raise DesignCompletenessError(f"cannot parse correct_start={value!r}")


def _check_group(by_cell, label, expected, scenarios, n_units):
    present = {(c[0], c[1], c[2]) for c in by_cell if (c[0], c[1], c[2]) in expected}
    missing = sorted(expected - present)
    if missing:
        raise DesignCompletenessError(
            f"missing {len(missing)} {label} cell(s), e.g. {missing[:3]}")
    for key in expected:
        cell_units = {c[3] for c in by_cell if (c[0], c[1], c[2]) == key}
        if len(cell_units) != n_units:
            raise DesignCompletenessError(
                f"{label} {key}: {len(cell_units)} unit(s), expected {n_units}")
        for unit in cell_units:
            if by_cell[(key[0], key[1], key[2], unit)] != set(scenarios):
                raise DesignCompletenessError(
                    f"{label} {(key[0], key[1], key[2], unit)}: scenario set differs from "
                    f"the fixed set of {len(scenarios)}")


def validate_design_completeness(rows, spec=DesignSpec()):
    """Raise DesignCompletenessError unless every pre-registered cell and scenario is present.
    The rho of Baseline 2 is ignored (one classification at rho_ref = 1.0 s)."""
    by_cell = {}
    valid_start = {}
    unexpected = sorted({int(float(r["D"])) for r in rows} - set(spec.densities))
    if unexpected:
        raise DesignCompletenessError(
            f"unexpected density D outside the design: {unexpected}")

    for r in rows:
        mode = str(r["mode"]).strip()
        D = int(float(r["D"]))
        unit = str(r["unit"]).strip()
        seed = int(float(r["episode_seed"]))
        rho = None if mode == spec.b2_arm else _rho_key(r.get("rho"))
        cell = (mode, D, rho, unit)

        episodes = by_cell.setdefault(cell, set())
        if seed in episodes:
            raise DesignCompletenessError(
                f"duplicate row for cell={cell}, episode_seed={seed}")
        episodes.add(seed)

        vs = _as_bool(r["correct_start"])
        key = (D, seed)
        if key in valid_start and valid_start[key] != vs:
            raise DesignCompletenessError(
                f"inconsistent correct_start for (D={D}, episode_seed={seed}) across cells")
        valid_start[key] = vs

    common = {(m, D, round(rho, 6)) for m in spec.h1a_arms for D in spec.densities
              for rho in spec.h1a_rhos}
    _check_group(by_cell, "H1a", common, spec.common_scenarios, spec.n_units)

    h1b = {(spec.b2_arm, D, None) for D in spec.densities}
    _check_group(by_cell, "H1b", h1b, spec.b2_scenarios, spec.n_units)

    h23 = {(m, D, round(rho, 6)) for m in spec.h1a_arms for D in spec.densities
           for rho in spec.h23_rhos}
    _check_group(by_cell, "H2/H3", h23, spec.common_scenarios, spec.n_units)
    return None
