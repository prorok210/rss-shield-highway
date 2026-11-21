"""Regression test for remark 4: NaN, +/-inf and bool must not pass the zone validator."""
import math

import pytest

from rss_shield.shield import (MAIN_SERIES, ShieldConfig, ZoneError, complete_zone,
                               validate_zone)


def test_nan_and_infinities_are_rejected():
    need = complete_zone(1.0)
    for bad in (float("nan"), float("inf"), float("-inf"), math.nan, math.inf):
        with pytest.raises(ZoneError):
            validate_zone(bad, need)
        with pytest.raises(ZoneError):
            ShieldConfig(rho=1.0, zone=bad)


def test_bool_and_small_zone_are_rejected():
    with pytest.raises(ZoneError):
        ShieldConfig(rho=1.0, zone=True)
    with pytest.raises(ZoneError):
        ShieldConfig(rho=1.0, zone=False)
    with pytest.raises(ZoneError):
        ShieldConfig(rho=1.0, zone=complete_zone(1.0) - 1.0)
    assert ShieldConfig(rho=1.0, zone=complete_zone(1.0)).zone == pytest.approx(complete_zone(1.0))
    assert ShieldConfig(rho=1.0, **MAIN_SERIES).zone is None


def test_other_config_numbers_are_finite():
    for kw in (dict(rho=float("nan")), dict(rho=float("inf")), dict(rho=True),
               dict(rho=0.6, lc_horizon=float("nan")), dict(rho=0.6, lc_horizon=0.0),
               dict(rho=0.6, eps=float("inf")), dict(rho=0.6, eps=-1.0)):
        with pytest.raises(ValueError):
            ShieldConfig(**kw)


def test_validate_zone_rejects_non_numeric():
    with pytest.raises(ZoneError):
        validate_zone("wide", 100.0)
    assert validate_zone(None, 100.0) is None
    assert validate_zone(100.0, 100.0) == 100.0
