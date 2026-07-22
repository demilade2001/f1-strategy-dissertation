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

V2_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_v2_full_peryear_stats.csv"
V3_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_v3_full_peryear_stats.csv"

PROB_SOURCE = "lap_level"
N_ITERATIONS = 10_000
LAMBDA_VALUE = 1.0
SEED = 42
BACKSTOP_FLOOR = 2

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

CHECKABLE_REALITY = {
    ("Italian Grand Prix", 2024): {
        "subject_driver": "PIA",
        "max_plausible_win_rate": 0.85,
        "reference": "Leclerc won one-stop; Piastri finished second",
    }
}


def emit(text: str = "") -> None:
    print(text, flush=True)


def _strategy_key(strategy: Mapping[str, object]) -> Tuple[str, Tuple[Tuple[int, str], ...]]:
    return (
        str(strategy["starting_compound"]),
        tuple((int(l), str(c)) for l, c in strategy.get("stops", [])),
    )


def _median_targets(df: pd.DataFrame) -> Dict[str, float]:
    targets: Dict[str, float] = {}
    if "shrink_target_locked_compound_median" in df.columns:
        src = df[df["shrink_target_locked_compound_median"].notna()].copy()
        for comp, grp in src.groupby("TyreCompound"):
            targets[str(comp)] = float(grp["shrink_target_locked_compound_median"].median())

    for comp in DRY_COMPOUNDS:
        if comp in targets:
            continue
        vals = df[df["TyreCompound"] == comp]["deg_rate_final_assigned"].dropna()
        targets[comp] = float(vals.median()) if not vals.empty else np.nan

    return targets


def _build_v3_from_v2(v2_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    targets = _median_targets(v2_df)

    v3 = v2_df.copy()
    if "fallback_trigger_reason_v3" not in v3.columns:
        base_reason = v3.get("fallback_trigger_reason", pd.Series([""] * len(v3), index=v3.index))
        v3["fallback_trigger_reason_v3"] = base_reason.fillna("").astype(str)
    if "deg_rate_pre_fallback_v3" not in v3.columns:
        v3["deg_rate_pre_fallback_v3"] = v3["deg_rate_final_assigned"]

    lock_rows = v3.merge(
        locked[["event_name", "year", "round", "archetype"]],
        left_on=["EventName", "Year"],
        right_on=["event_name", "year"],
        how="inner",
    ).copy()

    additional_rows: List[Dict[str, object]] = []
    all_negative_rows: List[Dict[str, object]] = []

    for row in lock_rows.itertuples(index=False):
        value = getattr(row, "deg_rate_final_assigned")
        if pd.isna(value) or float(value) >= 0.0:
            continue

        comp = str(row.TyreCompound)
        target = float(targets.get(comp, np.nan))
        prev_reason_raw = getattr(row, "fallback_trigger_reason", "")
        prev_reason = "" if pd.isna(prev_reason_raw) else str(prev_reason_raw).strip()
        combined_reason = prev_reason if prev_reason else ""
        if combined_reason:
            combined_reason = f"{combined_reason}|neg_slope"
        else:
            combined_reason = "neg_slope"

        mask = (
            (v3["EventName"] == row.EventName)
            & (v3["Year"] == int(row.Year))
            & (v3["TyreCompound"] == comp)
        )
        v3.loc[mask, "fallback_trigger_reason_v3"] = combined_reason
        v3.loc[mask, "shrink_target_locked_compound_median"] = target
        if np.isfinite(target):
            v3.loc[mask, "deg_rate_final_assigned"] = target

        row_info = {
            "EventName": row.EventName,
            "Year": int(row.Year),
            "Round": int(row.round),
            "Archetype": row.archetype,
            "TyreCompound": comp,
            "deg_rate_pre_fallback_v3": float(value),
            "deg_rate_post_fallback_v3": target,
            "prior_v2_reason": prev_reason,
            "trigger_reason_v3": "neg_slope",
            "model": str(row.model),
        }
        all_negative_rows.append(row_info)

        if not prev_reason:
            additional_rows.append(row_info)

    additional_df = pd.DataFrame(additional_rows).sort_values(
        ["EventName", "Year", "TyreCompound"]
    ).reset_index(drop=True) if additional_rows else pd.DataFrame(columns=[
        "EventName", "Year", "Round", "Archetype", "TyreCompound", "deg_rate_pre_fallback_v3",
        "deg_rate_post_fallback_v3", "prior_v2_reason", "trigger_reason_v3", "model"
    ])

    all_negative_df = pd.DataFrame(all_negative_rows).sort_values(
        ["EventName", "Year", "TyreCompound"]
    ).reset_index(drop=True) if all_negative_rows else pd.DataFrame(columns=additional_df.columns)

    return v3, additional_df, all_negative_df


def _load_race_state_with_deg(race: Mapping[str, object], deg_table: pd.DataFrame) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    event_name = str(state["laps"]["EventName"].dropna().iloc[0])
    state["deg_stats"] = deg_table[
        (deg_table["Year"] == int(race["year"]))
        & (deg_table["EventName"] == event_name)
    ].copy()
    return state


def _is_wet_race(actual_strategies: Mapping[str, Mapping[str, object]]) -> bool:
    for strategy in actual_strategies.values():
        start_comp = str(strategy.get("starting_compound", ""))
        if start_comp in WET_COMPOUNDS:
            return True
        for _, compound in strategy.get("stops", []):
            if str(compound) in WET_COMPOUNDS:
                return True
    return False


def _enumerate_subject_strategies(race_state: dict, min_stint_length_laps: int) -> dict:
    laps = race_state["laps"].copy()
    subject_driver = select_midfield_subject_driver(laps)
    actual_strategies = build_actual_strategies(laps)
    subject_actual = dict(actual_strategies[subject_driver])
    wet_race = _is_wet_race(actual_strategies)

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    base_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=wet_race,
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
        "actual_strategies": actual_strategies,
        "wet_race": wet_race,
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
    wet_race = _is_wet_race(actual)

    feasible = 0
    failed = []
    for driver, strategy in sorted(actual.items()):
        ok = is_strategy_feasible(
            race_length=race_length,
            starting_compound=str(strategy["starting_compound"]),
            is_wet_race=wet_race,
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
        "wet_race": wet_race,
        "feasible_drivers": int(feasible),
        "total_drivers": int(total),
        "feasibility_rate": float(feasible / total) if total else np.nan,
        "failed_drivers": failed,
    }


def _subject_stint_only_feasible(enum_ctx: dict, min_stint_length_laps: int) -> bool:
    strategy = enum_ctx["subject_actual"]
    return is_strategy_feasible(
        race_length=int(enum_ctx["race_length"]),
        starting_compound=str(strategy["starting_compound"]),
        is_wet_race=bool(enum_ctx["wet_race"]),
        stops=[(int(l), str(c)) for l, c in strategy.get("stops", [])],
        dry_compounds=DRY_COMPOUNDS,
        max_stops=MAX_STOPS,
        min_stint_length_laps=int(min_stint_length_laps),
    )


def main() -> None:
    emit("=" * 100)
    emit("Step 5g - Final combined fix")
    emit("=" * 100)
    emit(f"v2_input_path={V2_PATH}")
    emit(f"v3_output_path={V3_PATH}")
    emit(f"backstop_floor={BACKSTOP_FLOOR}")

    v2 = pd.read_csv(V2_PATH)
    v3, additional_neg, all_neg = _build_v3_from_v2(v2)
    V3_PATH.parent.mkdir(parents=True, exist_ok=True)
    v3.to_csv(V3_PATH, index=False)

    emit("")
    emit("=" * 100)
    emit("Step 1 - Extend shrinkage rule with negative-slope trigger")
    emit("=" * 100)
    emit(f"v2_rows={len(v2)}")
    emit(f"v3_rows={len(v3)}")
    emit(f"negative_slope_rows_locked_total={len(all_neg)}")
    emit(f"negative_slope_rows_newly_caught_beyond_v2={len(additional_neg)}")

    emit("negative_slope_all_locked_rows_begin")
    for row in all_neg.to_dict(orient="records"):
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("negative_slope_all_locked_rows_end")

    emit("negative_slope_newly_caught_beyond_v2_begin")
    for row in additional_neg.to_dict(orient="records"):
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("negative_slope_newly_caught_beyond_v2_end")

    british_soft_caught = False
    if not additional_neg.empty:
        british_soft_caught = bool(
            (
                (additional_neg["EventName"] == "British Grand Prix")
                & (additional_neg["Year"] == 2023)
                & (additional_neg["TyreCompound"] == "SOFT")
            ).any()
        )
    emit(f"british_gp_2023_soft_confirmed_newly_caught={british_soft_caught}")

    emit("")
    emit("=" * 100)
    emit("Step 2 - Wet-compound exemption verification")
    emit("=" * 100)
    wet_focus = [
        ("Monaco Grand Prix", 2023),
        ("Singapore Grand Prix", 2022),
        ("Japanese Grand Prix", 2022),
    ]
    for event_name, year in wet_focus:
        race = next(r for r in LOCKED_ARCHETYPE_RACES if r["event_name"] == event_name and int(r["year"]) == year)
        state = _load_race_state_with_deg(race, v3)
        enum_ctx = _enumerate_subject_strategies(state, min_stint_length_laps=BACKSTOP_FLOOR)
        feas = _actual_feasibility_rate(state, min_stint_length_laps=BACKSTOP_FLOOR)
        emit("wet_exemption_recheck=" + json.dumps({
            "race": race,
            "wet_race_detected": enum_ctx["wet_race"],
            "subject_driver": enum_ctx["subject_driver"],
            "subject_starting_compound": enum_ctx["subject_actual"]["starting_compound"],
            "subject_actual_strategy": enum_ctx["subject_actual"],
            "subject_strategy_count_after_wet_exemption": len(enum_ctx["strategies"]),
            "actual_feasibility_rate": feas["feasibility_rate"],
            "feasible_drivers": feas["feasible_drivers"],
            "total_drivers": feas["total_drivers"],
        }, sort_keys=True, default=str))

    emit("")
    emit("=" * 100)
    emit("Step 3 - Full 12-race rerun (argmax + win-rate scan + feasibility)")
    emit("=" * 100)

    race_results: List[Dict[str, object]] = []
    feasibility_rows: List[Dict[str, object]] = []
    high_win_rows: List[Dict[str, object]] = []

    for race in LOCKED_ARCHETYPE_RACES:
        emit(f"running_race event={race['event_name']} year={race['year']} round={race['round']}")
        state = _load_race_state_with_deg(race, v3)
        enum_ctx = _enumerate_subject_strategies(state, min_stint_length_laps=BACKSTOP_FLOOR)
        prepared = _prepare_race_context(state, enum_ctx)
        scored = _score_argmax(enum_ctx, prepared)

        win_rate_flag = bool(float(scored["best_win_rate"]) > 0.85)
        race_row = {
            "race": race,
            "subject_driver": enum_ctx["subject_driver"],
            "wet_race": enum_ctx["wet_race"],
            "subject_actual_stint_only_feasible": _subject_stint_only_feasible(enum_ctx, BACKSTOP_FLOOR),
            **scored,
            "win_rate_gt_0p85_flag": win_rate_flag,
        }
        race_results.append(race_row)
        if win_rate_flag:
            high_win_rows.append(race_row)

        emit("race_argmax_summary=" + json.dumps({
            "race": race,
            "subject_driver": enum_ctx["subject_driver"],
            "wet_race": enum_ctx["wet_race"],
            "strategy_count": scored["strategy_count"],
            "best_mean_points": scored["best_mean_points"],
            "best_win_rate": scored["best_win_rate"],
            "actual_subject_strategy_in_feasible_set": scored["actual_subject_strategy_in_feasible_set"],
            "subject_actual_stint_only_feasible": race_row["subject_actual_stint_only_feasible"],
            "win_rate_gt_0p85_flag": win_rate_flag,
        }, sort_keys=True, default=str))

        feas = _actual_feasibility_rate(state, min_stint_length_laps=BACKSTOP_FLOOR)
        feasibility_rows.append({"race": race, **feas})

    emit("all_race_argmax_table_begin")
    for row in race_results:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("all_race_argmax_table_end")

    emit("win_rate_flags_gt_0p85_begin")
    for row in high_win_rows:
        emit(json.dumps({
            "race": row["race"],
            "subject_driver": row["subject_driver"],
            "best_win_rate": row["best_win_rate"],
            "best_mean_points": row["best_mean_points"],
            "requires_explanation": True,
        }, sort_keys=True, default=str))
    emit("win_rate_flags_gt_0p85_end")

    emit("feasibility_table_begin")
    for row in feasibility_rows:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("feasibility_table_end")

    feas_df = pd.DataFrame(feasibility_rows)
    overall_feasible = int(feas_df["feasible_drivers"].sum())
    overall_total = int(feas_df["total_drivers"].sum())
    overall_rate = float(overall_feasible / overall_total) if overall_total else np.nan
    emit(f"overall_feasible_drivers={overall_feasible}")
    emit(f"overall_total_drivers={overall_total}")
    emit(f"overall_feasibility_rate={overall_rate:.6f}")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Belgian/PIA discrepancy reconciliation")
    emit("=" * 100)

    belgium = next(r for r in race_results if r["race"]["event_name"] == "Belgian Grand Prix" and int(r["race"]["year"]) == 2024)
    emit(
        "argmax_membership_definition=exact strategy key membership in the enumerated, grid-constrained candidate set (starting compound plus exact stop laps/compounds)."
    )
    emit(
        "step5_feasibility_definition=stint-length and rule-feasibility only via is_strategy_feasible, without requiring exact grid-lap membership in enumerated candidates."
    )
    emit("belgian_pia_reconciliation=" + json.dumps({
        "race": belgium["race"],
        "subject_driver": belgium["subject_driver"],
        "actual_subject_strategy_in_feasible_set": belgium["actual_subject_strategy_in_feasible_set"],
        "subject_actual_stint_only_feasible": belgium["subject_actual_stint_only_feasible"],
        "note": "These checks are intentionally different; false on exact membership can coexist with true on stint-only feasibility.",
    }, sort_keys=True, default=str))

    emit("")
    emit("=" * 100)
    emit("Step 5 - Stopping-bar evaluation")
    emit("=" * 100)

    overall_feasibility_pass = bool(overall_rate >= 0.80)

    fallback_reason_col = "fallback_trigger_reason_v3" if "fallback_trigger_reason_v3" in v3.columns else "fallback_trigger_reason"
    japan_rows = v3[(v3["EventName"] == "Japanese Grand Prix") & (v3["Year"] == 2022)].copy()
    japan_absent_only = False
    if not japan_rows.empty:
        reasons = japan_rows[fallback_reason_col].fillna("").astype(str)
        japan_absent_only = bool((reasons.str.contains("absent_key")).all())

    contradictions = []
    for row in race_results:
        race_key = (str(row["race"]["event_name"]), int(row["race"]["year"]))
        if race_key not in CHECKABLE_REALITY:
            continue
        policy = CHECKABLE_REALITY[race_key]
        if str(row["subject_driver"]) != str(policy["subject_driver"]):
            continue
        if float(row["best_win_rate"]) > float(policy["max_plausible_win_rate"]):
            contradictions.append(
                {
                    "race": row["race"],
                    "subject_driver": row["subject_driver"],
                    "best_win_rate": row["best_win_rate"],
                    "max_plausible_win_rate": policy["max_plausible_win_rate"],
                    "reference": policy["reference"],
                }
            )

    unexplained_high_win = []
    for row in high_win_rows:
        race_key = (str(row["race"]["event_name"]), int(row["race"]["year"]))
        if race_key == ("Japanese Grand Prix", 2022) and japan_absent_only:
            continue
        unexplained_high_win.append(
            {
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "best_win_rate": row["best_win_rate"],
                "best_mean_points": row["best_mean_points"],
            }
        )

    stopping_bar_pass = bool(overall_feasibility_pass and len(contradictions) == 0 and len(unexplained_high_win) == 0)

    emit(f"stopping_bar_overall_feasibility_ge_0p80={overall_feasibility_pass}")
    emit(f"stopping_bar_contradictions_count={len(contradictions)}")
    emit(f"stopping_bar_unexplained_high_win_count={len(unexplained_high_win)}")
    emit(f"stopping_bar_pass={stopping_bar_pass}")

    emit("documented_explanations_begin")
    emit(json.dumps({
        "race": {"event_name": "Japanese Grand Prix", "year": 2022},
        "explanation": "All dry compounds rest on absent-key fallback-only degradation values.",
        "applies": japan_absent_only,
    }, sort_keys=True, default=str))
    emit("documented_explanations_end")

    emit("checkable_real_result_contradictions_begin")
    for row in contradictions:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("checkable_real_result_contradictions_end")

    emit("residual_limitations_begin")
    for row in unexplained_high_win:
        emit(json.dumps({
            "type": "implausible_high_win_rate",
            "race": row["race"],
            "subject_driver": row["subject_driver"],
            "best_win_rate": row["best_win_rate"],
            "statement": "Residual limitation for Chapter 5 under current final combined fix; no further fix cycle opened in this step.",
        }, sort_keys=True, default=str))
    if contradictions:
        for row in contradictions:
            emit(json.dumps({
                "type": "checkable_real_result_contradiction",
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "best_win_rate": row["best_win_rate"],
                "reference": row["reference"],
                "statement": "Residual limitation for Chapter 5 under current final combined fix; no further fix cycle opened in this step.",
            }, sort_keys=True, default=str))
    emit("residual_limitations_end")


if __name__ == "__main__":
    main()
