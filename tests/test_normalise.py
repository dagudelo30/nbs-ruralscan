from __future__ import annotations

import numpy as np
import pytest

from nbs_ruralscan.runtime.normalise import (
    linear_decay,
    log_transform,
    min_max,
    normalise,
    percentile_clip,
)


def test_min_max_basic():
    out = min_max(np.array([0, 5, 10]))
    assert list(out) == [0.0, 0.5, 1.0]


def test_percentile_clip_bounds():
    x = np.array([-100, 0, 50, 100, 10000])
    out = percentile_clip(x, p_low=10, p_high=90)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_log_transform_compresses_tail():
    x = np.array([0, 1, 1000])
    out = log_transform(x)
    assert out[-1] == 1.0
    assert out[0] == 0.0


def test_linear_decay_shape():
    out = linear_decay(np.array([0, 5, 10]), suitable_max=0, unsuitable_min=10)
    assert out[0] == 1.0
    assert out[2] == 0.0
    assert out[1] == pytest.approx(0.5)


def test_normalise_negative_risk_inverts():
    x = np.array([0, 5, 10])
    positive = normalise(x, "min_max", {}, directionality="positive_risk")
    negative = normalise(x, "min_max", {}, directionality="negative_risk")
    assert np.allclose(negative, 1.0 - positive)


def test_normalise_unknown_method_raises():
    with pytest.raises(KeyError):
        normalise(np.array([1.0]), "not_a_method", {})
