from __future__ import annotations

from typing import Dict, Mapping


def _draw_uniform(rng) -> float:
    if hasattr(rng, "random") and callable(rng.random):
        return float(rng.random())
    if hasattr(rng, "random_sample") and callable(rng.random_sample):
        return float(rng.random_sample())
    if callable(rng):
        return float(rng())
    raise TypeError("rng must expose a random(), random_sample(), or callable interface")


def sample_caution_schedule(prob_by_lap: Mapping[int, float], rng) -> Dict[int, bool]:
    """Draw a Bernoulli caution schedule from per-lap probabilities.

    The caller owns the probability source. This function is agnostic to whether
    the inputs came from lap-level XGBoost predictions, a flat circuit prior, or
    any other per-lap probability dictionary.
    """

    schedule: Dict[int, bool] = {}
    for lap_number in sorted(prob_by_lap.keys()):
        probability = prob_by_lap[lap_number]
        if probability is None:
            schedule[int(lap_number)] = False
            continue
        probability = float(probability)
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"Probability for lap {lap_number} must be in [0, 1]; got {probability}")
        schedule[int(lap_number)] = _draw_uniform(rng) < probability
    return schedule