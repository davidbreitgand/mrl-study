"""Tests for the Weibull inter-arrival extension to open_loop_staged.

The costguard fork of the harness supports only exponential gaps (Poisson
arrivals) in the open_loop_staged driver. The mrl_study fork adds an
optional `weibull_shape` field on ArrivalStage; when set, the driver draws
Weibull-shaped inter-arrival gaps with scale chosen so E[gap] = 1/lambda.

These tests exercise:
  1. The ArrivalStage schema accepts weibull_shape.
  2. Weibull draws with matching mean behave as expected on a large sample.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from pricing_reversal_harness.config import ArrivalStage


def test_arrival_stage_accepts_weibull_shape():
    stage = ArrivalStage(**{"lambda": 10.0, "rows": 100, "weibull_shape": 0.5})
    assert stage.lambda_ == 10.0
    assert stage.rows == 100
    assert stage.weibull_shape == 0.5


def test_arrival_stage_weibull_shape_optional():
    stage = ArrivalStage(**{"lambda": 10.0, "rows": 100})
    assert stage.weibull_shape is None


def test_arrival_stage_weibull_shape_must_be_positive():
    with pytest.raises(ValueError):
        ArrivalStage(**{"lambda": 10.0, "rows": 100, "weibull_shape": 0.0})
    with pytest.raises(ValueError):
        ArrivalStage(**{"lambda": 10.0, "rows": 100, "weibull_shape": -0.5})


def _draw_weibull_gaps(lam: float, shape: float, n: int, seed: int) -> list[float]:
    """Mirror the runner.py Weibull-branch scale computation exactly."""
    rng = random.Random(seed)
    scale = 1.0 / (lam * math.gamma(1.0 + 1.0 / shape))
    return [rng.weibullvariate(scale, shape) for _ in range(n)]


@pytest.mark.parametrize("shape", [0.3, 0.5, 1.0, 2.0])
def test_weibull_gap_mean_matches_one_over_lambda(shape: float):
    """The scale used in the runner keeps E[gap] = 1/lambda across shapes.

    With lambda=10 and a 10k-sample draw, the sample mean stays within a
    few percent of 0.1s regardless of shape, so offered load is held fixed
    while only the burstiness (CV) varies.
    """
    lam = 10.0
    gaps = _draw_weibull_gaps(lam, shape, n=10_000, seed=1234)
    sample_mean = statistics.fmean(gaps)
    expected_mean = 1.0 / lam
    # Weibull(0.3) is heavy-tailed and 10k samples give ~5% error; looser bound.
    rel_tol = 0.10 if shape < 0.5 else 0.05
    assert abs(sample_mean - expected_mean) / expected_mean < rel_tol, (
        f"shape={shape}: sample mean {sample_mean:.4f} vs expected {expected_mean:.4f}"
    )


def test_weibull_gap_cv_scales_with_shape():
    """Under Weibull, CV increases (sharply) as shape decreases.

    Analytical CV is sqrt(Gamma(1+2/k)/Gamma(1+1/k)^2 - 1). At k=1 this
    equals 1.0 (exponential); at k=0.5 it is ~2.24; at k=0.3 it is
    ~5.41. Under 20k samples the sample CV of the heavy-tailed k=0.3
    case shows real dispersion around its analytical value, so the
    bound below is intentionally loose.
    """
    lam = 10.0
    gaps_05 = _draw_weibull_gaps(lam, 0.5, n=20_000, seed=1)
    gaps_03 = _draw_weibull_gaps(lam, 0.3, n=20_000, seed=2)
    cv_05 = statistics.pstdev(gaps_05) / statistics.fmean(gaps_05)
    cv_03 = statistics.pstdev(gaps_03) / statistics.fmean(gaps_03)
    assert 1.8 < cv_05 < 2.8, f"Weibull-0.5 sample CV {cv_05:.3f} outside plausible range"
    assert 3.5 < cv_03 < 12.0, f"Weibull-0.3 sample CV {cv_03:.3f} outside plausible range"
    assert cv_03 > cv_05, "Weibull-0.3 must be burstier than Weibull-0.5"
