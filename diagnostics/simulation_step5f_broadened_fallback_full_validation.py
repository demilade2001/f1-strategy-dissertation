from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.config import GRID_SPACING, LOCKED_ARCHETYPE_RACES, MAX_STOPS
from src.simulation.monte_carlo import build_actual_strategies, select_midfield_subject_driver
from src.simulation.optimiser import (
    ARGMAX_BATCH_SIZE,
    _build_caution_schedule_matrix,
    _precompute_subject_inputs,
    _resolve_probability_by_source,
)
from src.simulation.race_model_vectorized import evaluate_strategy_batch, rank_and_score_batch
from src.simulation.race_state import load_race_state
from src.simulation.strategy import enumerate_feasible_strategies, is_strategy_feasible

V1_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_full_peryear_stats.csv"
V2_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_v2_full_peryear_stats.csv"

PROB_SOURCE = "lap_level"
N_ITERATIONS = 10_000
LAMBDA_VALUE = 1.0
SEED = 42
BACKSTOP_FLOOR = 2

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
UNRELIABLE_MODELS = {"missing", "insufficient", "invalid_data"}


def emit(text: str = "") -> None:
    print(text, flush=True)


def _strategy_key(strategy: Mapping[str, object]) -> Tuple[str, Tuple[Tuple[int, str], ...]]:
    return (
        str(strategy["starting_compound"]),
        tuple((int(l), str(c)) for l, c in strategy.get("stops", [])),
    )


def _median_targets(v1_df: pd.DataFrame) -> Dict[str, float]:
    # Prefer existing computed targets from step5d.
    targets: Dict[str, float] = {}
    target_col = "shrink_target_locked_compound_median"
    if target_col in v1_df.columns:
        sub = v1_df[v1_df[target_col].notna()].copy()
        for comp, grp in sub.groupby("TyreCompound"):
            targets[str(comp)] = float(grp[target_col].median())

    # Fallback if any compound target absent: derive from reliable locked-race fits.
    locked = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    lock_rows = v1_df.merge(
        locked[["event_name", "year"]],
        left_on=["EventName", "Year"],
        right_on=["event_name", "year"],
        how="inner",
    )

    if "is_eligible_fit" in lock_rows.columns:
        reliable = lock_rows[lock_rows["is_eligible_fit"].eq(True)].copy()
    else:
        reliable = lock_rows.copy()
        reliable = reliable[~reliable["low_sample"].fillna(False)]
        reliable = reliable[~reliable["model"].isin(UNRELIABLE_MODELS)]
        reliable = reliable[reliable["deg_rate_final_assigned"].notna()]

    if "is_thin_r2" in reliable.columns:
        reliable = reliable[~reliable["is_thin_r2"].fillna(False)]

    for comp in DRY_COMPOUNDS:
        if comp in targets:
            continue
        vals = reliable[reliable["TyreCompound"] == comp]["deg_rate_final_assigned"].dropna()
        targets[comp] = float(vals.median()) if not vals.empty else np.nan

    return targets


def _broaden_fallback(v1_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    locked = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    targets = _median_targets(v1_df)

    v2 = v1_df.copy()
    if "fallback_trigger_reason" not in v2.columns:
        v2["fallback_trigger_reason"] = ""
    if "deg_rate_pre_fallback_v2" not in v2.columns:
        v2["deg_rate_pre_fallback_v2"] = v2["deg_rate_final_assigned"]

    affected_rows: List[Dict[str, object]] = []

    # Existing rows in locked races: thin-R2 OR missing/insufficient/invalid_data.
    lock_rows = v2.merge(
        locked[["event_name", "year", "round", "archetype"]],
        left_on=["EventName", "Year"],
        right_on=["event_name", "year"],
        how="inner",
    ).copy()

    for row in lock_rows.itertuples(index=False):
        model = str(row.model)
        thin = bool(getattr(row, "is_thin_r2", False))
        trigger_reason = None
        if thin:
            trigger_reason = "thin_r2"
        elif model in UNRELIABLE_MODELS:
            trigger_reason = f"status_{model}"

        if trigger_reason is None:
            continue

        comp = str(row.TyreCompound)
        target = float(targets.get(comp, np.nan))
        mask = (
            (v2["EventName"] == row.EventName)
            & (v2["Year"] == int(row.Year))
            & (v2["TyreCompound"] == comp)
        )
        v2.loc[mask, "fallback_trigger_reason"] = trigger_reason
        v2.loc[mask, "shrink_target_locked_compound_median"] = target
        if np.isfinite(target):
            v2.loc[mask, "deg_rate_final_assigned"] = target

        affected_rows.append(
            {
                "EventName": row.EventName,
                "Year": int(row.Year),
                "Round": int(row.round),
                "Archetype": row.archetype,
                "TyreCompound": comp,
                "trigger_reason": trigger_reason,
                "deg_rate_pre_fallback_v2": float(row.deg_rate_final_assigned)
                if pd.notna(row.deg_rate_final_assigned)
                else np.nan,
                "deg_rate_post_fallback_v2": target,
                "model": model,
            }
        )

    # Absent keys in locked races: add synthetic rows and fallback.
    existing_keys = {
        (str(r.EventName), int(r.Year), str(r.TyreCompound))
        for r in v2.itertuples(index=False)
        if pd.notna(r.TyreCompound)
    }

    synthetic_rows: List[Dict[str, object]] = []
    for race in LOCKED_ARCHETYPE_RACES:
        event_name = str(race["event_name"])
        year = int(race["year"])
        round_num = int(race["round"])
        archetype = str(race["archetype"])

        for comp in DRY_COMPOUNDS:
            key = (event_name, year, comp)
            if key in existing_keys:
                continue

            target = float(targets.get(comp, np.nan))
            synthetic_rows.append(
                {
                    "TyreCompound": comp,
                    "EventName": event_name,
                    "Year": year,
                    "n_laps": 0,
                    "model": "absent_key",
                    "deg_rate_internal": np.nan,
                    "deg_rate_final_assigned": target,
                    "r2_linear": np.nan,
                    "r2_quadratic": np.nan,
                    "low_sample": True,
                    "selected_r2": np.nan,
                    "is_eligible_fit": False,
                    "shrink_target_locked_compound_median": target,
                    "is_thin_r2": False,
                    "deg_rate_pre_shrink": np.nan,
                    "fallback_trigger_reason": "absent_key",
                    "deg_rate_pre_fallback_v2": np.nan,
                }
            )

            affected_rows.append(
                {
                    "EventName": event_name,
                    "Year": year,
                    "Round": round_num,
                    "Archetype": archetype,
                    "TyreCompound": comp,
                    "trigger_reason": "absent_key",
                    "deg_rate_pre_fallback_v2": np.nan,
                    "deg_rate_post_fallback_v2": target,
                    "model": "absent_key",
                }
            )

    if synthetic_rows:
        v2 = pd.concat([v2, pd.DataFrame(synthetic_rows)], ignore_index=True)

    affected_df = pd.DataFrame(affected_rows)
    affected_df = affected_df.sort_values(["EventName", "Year", "TyreCompound"]).reset_index(drop=True)

    return v2, affected_df


def _load_race_state_with_deg(race: Mapping[str, object], deg_table: pd.DataFrame) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    event_name = str(state["laps"]["EventName"].dropna().iloc[0])
    state["deg_stats"] = deg_table[
        (deg_table["Year"] == int(race["year"]))
        & (deg_table["EventName"] == event_name)
    ].copy()
    return state


def _enumerate_subject_strategies(race_state: dict, min_stint_length_laps: int) -> dict:
    laps = race_state["laps"].copy()
    subject_driver = select_midfield_subject_driver(laps)
    actual_strategies = build_actual_strategies(laps)
    subject_actual = dict(actual_strategies[subject_driver])

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    base_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=False,
        max_stops=MAX_STOPS,
        min_stint_length_laps=int(min_stint_length_laps),
    )
    if hasattr(base_space, "materialize"):
        materialized = base_space.materialize(limit=2_000_000)
    else:
        materialized = list(base_space)

    grid_candidate_laps = set(range(1, race_length, GRID_SPACING))
    filtered = [
        s for s in materialized if all(int(lap) in grid_candidate_laps for lap, _ in s.get("stops", []))
    ]

    enriched = []
    for s in filtered:
        row = dict(s)
        row.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
        row.setdefault("year", int(subject_actual["year"]))
        row.setdefault("round", int(subject_actual["round"]))
        row.setdefault("event_name", str(subject_actual["event_name"]))
        enriched.append(row)

    return {
        "subject_driver": subject_driver,
        "subject_actual": subject_actual,
        "race_length": race_length,
        "strategies": enriched,
    }


def _prepare_race_context(race_state: dict, enum_ctx: dict) -> dict:
    prob_by_lap = _resolve_probability_by_source(race_state, PROB_SOURCE)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, N_ITERATIONS, np.random.default_rng(SEED))
    precomputed = _precompute_subject_inputs(
        race_state=race_state,
        subject_driver=enum_ctx["subject_driver"],
        caution_schedules=caution_schedules,
        lambda_=LAMBDA_VALUE,
    )
    return {"caution_schedules": caution_schedules, "precomputed": precomputed}


def _score_argmax(enum_ctx: dict, prepared_ctx: dict) -> dict:
    strategies = enum_ctx["strategies"]
    mean_points = np.zeros(len(strategies), dtype=np.float64)
    std_points = np.zeros(len(strategies), dtype=np.float64)
    mean_rank = np.zeros(len(strategies), dtype=np.float64)

    n_batches = (len(strategies) + ARGMAX_BATCH_SIZE - 1) // ARGMAX_BATCH_SIZE
    for batch_idx, start in enumerate(range(0, len(strategies), ARGMAX_BATCH_SIZE), start=1):
        end = min(start + ARGMAX_BATCH_SIZE, len(strategies))
        batch = strategies[start:end]
        subject_times = evaluate_strategy_batch(
            subject_strategies=batch,
            caution_schedules=prepared_ctx["caution_schedules"],
            degradation_lookup=prepared_ctx["precomputed"]["degradation_lookup"],
            lambda_=LAMBDA_VALUE,
            baseline_pace_s=float(enum_ctx["subject_actual"]["baseline_pace_s"]),
            pit_loss_s=float(prepared_ctx["precomputed"]["pit_loss_s"]),
            caution_pit_loss_s=5.0,
            caution_pace_ratio=1.3162357008284764,
        )
        scored = rank_and_score_batch(subject_times, prepared_ctx["precomputed"]["rival_times"])
        mean_points[start:end] = np.asarray(scored["mean_points"], dtype=np.float64)
        std_points[start:end] = np.asarray(scored["std_points"], dtype=np.float64)
        mean_rank[start:end] = np.asarray(scored["mean_rank"], dtype=np.float64)

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == n_batches:
            emit(
                f"batch_progress race={enum_ctx['subject_actual']['event_name']} "
                f"batch={batch_idx}/{n_batches} scored={end}/{len(strategies)}"
            )

    first_stints = np.array(
        [int(s["stops"][0][0]) if s.get("stops") else int(enum_ctx["race_length"]) for s in strategies]
    )
    dist = pd.Series(first_stints).value_counts().sort_index()

    best_idx = int(np.argmax(mean_points))
    best_strategy = strategies[best_idx]
    top_k = min(100, len(strategies))
    top_idx = np.argsort(-mean_points)[:top_k]
    top_short_count = int((first_stints[top_idx] <= 3).sum())
    top_short_frac = float(top_short_count / top_k)

    actual_key = _strategy_key(enum_ctx["subject_actual"])
    feasible_keys = {_strategy_key(s) for s in strategies}

    best_times = evaluate_strategy_batch(
        subject_strategies=[best_strategy],
        caution_schedules=prepared_ctx["caution_schedules"],
        degradation_lookup=prepared_ctx["precomputed"]["degradation_lookup"],
        lambda_=LAMBDA_VALUE,
        baseline_pace_s=float(enum_ctx["subject_actual"]["baseline_pace_s"]),
        pit_loss_s=float(prepared_ctx["precomputed"]["pit_loss_s"]),
        caution_pit_loss_s=5.0,
        caution_pace_ratio=1.3162357008284764,
    )
    best_scored = rank_and_score_batch(best_times, prepared_ctx["precomputed"]["rival_times"])
    rank_matrix = np.asarray(best_scored["rank_matrix"][0], dtype=np.int16)
    win_rate = float(np.mean(rank_matrix == 1))

    return {
        "strategy_count": int(len(strategies)),
        "best_strategy": best_strategy,
        "best_mean_points": float(mean_points[best_idx]),
        "best_std_points": float(std_points[best_idx]),
        "best_mean_rank": float(mean_rank[best_idx]),
        "best_win_rate": win_rate,
        "best_first_stint_length": int(first_stints[best_idx]),
        "top_k": int(top_k),
        "top100_short_stint_fraction": top_short_frac if top_k >= 100 else None,
        "top_short_stint_leq3_count": int(top_short_count),
        "actual_subject_strategy_in_feasible_set": bool(actual_key in feasible_keys),
        "first_stint_distribution": [
            {
                "first_stint_length": int(k),
                "count": int(v),
                "share": float(v / len(first_stints)),
            }
            for k, v in dist.items()
        ],
    }


def _actual_feasibility_rate(race_state: dict, min_stint_length_laps: int) -> dict:
    laps = race_state["laps"].copy()
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    actual = build_actual_strategies(laps)

    feasible = 0
    failed = []
    for driver, strategy in sorted(actual.items()):
        ok = is_strategy_feasible(
            race_length=race_length,
            starting_compound=str(strategy["starting_compound"]),
            is_wet_race=False,
            stops=[(int(l), str(c)) for l, c in strategy.get("stops", [])],
            dry_compounds=DRY_COMPOUNDS,
            max_stops=MAX_STOPS,
            min_stint_length_laps=int(min_stint_length_laps),
        )
        if ok:
            feasible += 1
        else:
            failed.append(driver)

    total = len(actual)
    return {
        "feasible_drivers": int(feasible),
        "total_drivers": int(total),
        "feasibility_rate": float(feasible / total) if total else np.nan,
        "failed_drivers": failed,
    }


def main() -> None:
    emit("=" * 100)
    emit("Step 5f - Broadened fallback full validation")
    emit("=" * 100)
    emit(f"v1_input_path={V1_PATH}")
    emit(f"v2_output_path={V2_PATH}")
    emit(f"backstop_floor={BACKSTOP_FLOOR}")

    v1 = pd.read_csv(V1_PATH)
    v2, affected = _broaden_fallback(v1)
    V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    v2.to_csv(V2_PATH, index=False)

    emit("")
    emit("=" * 100)
    emit("Step 1 - Broadened fallback trigger")
    emit("=" * 100)
    emit(f"v1_rows={len(v1)}")
    emit(f"v2_rows={len(v2)}")
    emit(f"fallback_triggered_rows_locked={len(affected)}")

    emit("")
    emit("=" * 100)
    emit("Step 2 - Locked race affected compounds and Japanese GP status")
    emit("=" * 100)

    emit("locked_race_fallback_table_begin")
    for row in affected.to_dict(orient="records"):
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("locked_race_fallback_table_end")

    japan = affected[(affected["EventName"] == "Japanese Grand Prix") & (affected["Year"] == 2022)]
    japan_compounds = sorted(japan["TyreCompound"].tolist())
    japan_all_fallback = set(japan_compounds) == set(DRY_COMPOUNDS)
    emit(
        "japanese_gp_2022_fallback_status="
        + json.dumps(
            {
                "compounds_with_fallback": japan_compounds,
                "all_three_compounds_fallback": japan_all_fallback,
                "reasons": sorted(japan["trigger_reason"].unique().tolist()),
            },
            sort_keys=True,
        )
    )
    emit(
        "japanese_gp_2022_limitation=Japan 2022 rests entirely on fallback values rather than race-specific fitted degradation rates; this remains a documented limitation regardless of downstream metrics."
    )

    emit("")
    emit("=" * 100)
    emit("Step 3 - Full 12-race argmax revalidation with floor=2")
    emit("=" * 100)

    race_results: List[Dict[str, object]] = []
    feasibility_rows: List[Dict[str, object]] = []

    for race in LOCKED_ARCHETYPE_RACES:
        emit(f"running_race event={race['event_name']} year={race['year']} round={race['round']}")
        state = _load_race_state_with_deg(race, v2)
        enum_ctx = _enumerate_subject_strategies(state, min_stint_length_laps=BACKSTOP_FLOOR)
        prepared = _prepare_race_context(state, enum_ctx)
        scored = _score_argmax(enum_ctx, prepared)

        result_row = {
            "race": race,
            "subject_driver": enum_ctx["subject_driver"],
            **scored,
        }
        race_results.append(result_row)

        emit("race_argmax_summary=" + json.dumps({
            "race": race,
            "subject_driver": enum_ctx["subject_driver"],
            "strategy_count": scored["strategy_count"],
            "best_strategy": scored["best_strategy"],
            "best_mean_points": scored["best_mean_points"],
            "best_win_rate": scored["best_win_rate"],
            "top100_short_stint_fraction": scored["top100_short_stint_fraction"],
            "actual_subject_strategy_in_feasible_set": scored["actual_subject_strategy_in_feasible_set"],
        }, sort_keys=True, default=str))

        emit("first_stint_distribution_begin")
        for row in scored["first_stint_distribution"]:
            emit(json.dumps(row, sort_keys=True))
        emit("first_stint_distribution_end")

        feas = _actual_feasibility_rate(state, min_stint_length_laps=BACKSTOP_FLOOR)
        feas_row = {
            "race": race,
            **feas,
        }
        feasibility_rows.append(feas_row)

    emit("")
    emit("all_race_argmax_table_begin")
    for row in race_results:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("all_race_argmax_table_end")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Italy plausibility check vs confirmed real result")
    emit("=" * 100)

    italy_row = next(
        r for r in race_results if r["race"]["event_name"] == "Italian Grand Prix" and int(r["race"]["year"]) == 2024
    )
    emit("italy_result_v2_floor2=" + json.dumps({
        "argmax_strategy": italy_row["best_strategy"],
        "mean_points": italy_row["best_mean_points"],
        "win_rate": italy_row["best_win_rate"],
        "real_world_reference": {
            "winner": "Charles Leclerc",
            "piastri_finish": 2,
            "winner_strategy": "one-stop",
        },
    }, sort_keys=True, default=str))

    italy_red_flag = float(italy_row["best_win_rate"]) >= 0.90
    emit(f"italy_win_rate_red_flag_ge_90pct={italy_red_flag}")
    if italy_red_flag:
        emit("italy_plausibility_statement=Residual Italy win-rate remains implausibly high relative to the confirmed real result; further explanation or model correction is required.")
    else:
        emit("italy_plausibility_statement=Italy win-rate is no longer in the >=90% red-flag zone relative to the confirmed real result.")

    emit("")
    emit("=" * 100)
    emit("Step 5 - Actual-strategy feasibility across all 12 races")
    emit("=" * 100)

    emit("feasibility_table_begin")
    for row in feasibility_rows:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("feasibility_table_end")

    feas_df = pd.DataFrame(feasibility_rows)
    overall_feasible = int(feas_df["feasible_drivers"].sum())
    overall_total = int(feas_df["total_drivers"].sum())
    overall_rate = float(overall_feasible / overall_total) if overall_total else np.nan
    per_race_threshold_ok = bool((feas_df["feasibility_rate"] >= 0.80).all())
    overall_threshold_ok = bool(overall_rate >= 0.80)

    emit(f"overall_feasible_drivers={overall_feasible}")
    emit(f"overall_total_drivers={overall_total}")
    emit(f"overall_feasibility_rate={overall_rate:.6f}")
    emit(f"per_race_threshold_80pct_pass={per_race_threshold_ok}")
    emit(f"overall_threshold_80pct_pass={overall_threshold_ok}")


if __name__ == "__main__":
    main()
