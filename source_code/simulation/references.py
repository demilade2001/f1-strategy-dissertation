"""Reference constructors for Phase 3 bias decomposition.

These helpers expose common interfaces for three references:
- Conservatism: parameter-free latest legal first-stop lap in dry races.
- Anchoring: thin wrapper around cached trigger-level anchoring outputs.
- SC underweighting: cache-first race-level comparison of lap-level vs static-prior optima.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from .config import MIN_STINT_LENGTH_LAPS


WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def _coalesce(mapping: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _normalise_compound(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def r_conservatism(race_state: Mapping[str, Any], subject_driver: str) -> Dict[str, Any]:
    """Build the Conservatism reference.

    R is the latest lap where a legal two-compound finish remains feasible under
    the global stint floor. This reference is parameter-free and does not depend
    on strategic scoring. Wet races are flagged as exempt.
    """

    race_length = _coalesce(
        race_state,
        ["race_length_laps", "race_length", "total_laps", "n_laps"],
    )
    if race_length is None:
        raise ValueError("race_state is missing race length (race_length_laps/total_laps)")

    subject_info = race_state.get("subject", {})
    if isinstance(subject_info, Mapping):
        final_compound = _coalesce(
            subject_info,
            ["actual_final_compound", "final_compound", "compound_final_stint"],
        )
    else:
        final_compound = None

    if final_compound is None:
        final_compound = _coalesce(
            race_state,
            [
                "actual_final_compound",
                "subject_final_compound",
                "final_compound",
                "driver_final_compound",
            ],
        )

    final_compound_norm = _normalise_compound(final_compound)
    wet_exempt = final_compound_norm in WET_COMPOUNDS
    reference_lap = int(race_length) - int(MIN_STINT_LENGTH_LAPS)

    return {
        "bias": "conservatism",
        "subject_driver": subject_driver,
        "reference_kind": "lap",
        "reference_lap": reference_lap,
        "reference_mean_points": None,
        "wet_exempt": wet_exempt,
        "final_compound": final_compound_norm,
        "stint_floor_laps": int(MIN_STINT_LENGTH_LAPS),
        "note": (
            "Wet-exempt race flagged (final compound INTERMEDIATE/WET); dry-race formula retained for audit only."
            if wet_exempt
            else None
        ),
    }


def r_anchoring(
    race_state: Mapping[str, Any],
    subject_driver: str,
    trigger_lap: int,
    cached_event: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Expose step6b anchoring references under a common interface.

    This wrapper is intentionally cache-first and does not create new strategy
    pools or run new Monte Carlo scoring. It expects a cached event row from
    step6b (or an equivalent mapping embedded in race_state).
    """

    event = cached_event
    if event is None:
        by_lap = race_state.get("benchmark_r_anchoring_by_trigger_lap", {})
        if isinstance(by_lap, Mapping):
            event = by_lap.get(trigger_lap)

    if not isinstance(event, Mapping):
        return {
            "bias": "anchoring",
            "subject_driver": subject_driver,
            "trigger_lap": trigger_lap,
            "computable": False,
            "reference_kind": "strategy_mean_points",
            "reference_mean_points": None,
            "reference_strategy": None,
            "added_lap_for_r": trigger_lap + 1,
            "note": "No cached step6b anchoring row found for trigger lap.",
        }

    reference_strategy = event.get("strategy")
    reference_mean_points = event.get("mean_points")
    if reference_mean_points is None:
        reference_mean_points = event.get("r_anchoring_mean_points")

    return {
        "bias": "anchoring",
        "subject_driver": subject_driver,
        "trigger_lap": trigger_lap,
        "computable": bool(event.get("computable", reference_mean_points is not None)),
        "reference_kind": "strategy_mean_points",
        "reference_mean_points": reference_mean_points,
        "reference_strategy": reference_strategy,
        "added_lap_for_r": event.get("added_lap_for_r", trigger_lap + 1),
        "r_pool_count": event.get("r_pool_count"),
        "source": "step6b_cached",
    }


def r_sc(
    race_state: Mapping[str, Any],
    subject_driver: str,
    cache_lookup: Optional[Callable[[Mapping[str, Any], str], Optional[Mapping[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Build the SC-underweighting reference from cached race-level pilot values.

    Expected cached values:
    - B: lap-level, unconditioned argmax strategy mean points.
    - R_SC: static-prior, unconditioned argmax strategy mean points.
    """

    cached: Mapping[str, Any] = race_state
    if cache_lookup is not None:
        looked_up = cache_lookup(race_state, subject_driver)
        if isinstance(looked_up, Mapping):
            cached = looked_up

    b_mean_points = _coalesce(
        cached,
        [
            "b_lap_level_mean_points",
            "benchmark_b_mean_points",
            "b_mean_points",
        ],
    )
    r_sc_mean_points = _coalesce(
        cached,
        [
            "r_sc_mean_points",
            "r_static_prior_mean_points",
            "static_prior_mean_points",
        ],
    )

    return {
        "bias": "sc_underweighting",
        "subject_driver": subject_driver,
        "reference_kind": "strategy_mean_points",
        "b_lap_level_mean_points": b_mean_points,
        "reference_mean_points": r_sc_mean_points,
        "source": "cached_pilot",
        "computable": (b_mean_points is not None and r_sc_mean_points is not None),
    }
