from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	 sys.path.insert(0, str(ROOT))

from source_code.simulation.config import CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S, LOCKED_ARCHETYPE_RACES
from source_code.simulation.monte_carlo import build_actual_strategies, select_midfield_subject_driver
from source_code.simulation.optimiser import (
	ARGMAX_BATCH_SIZE,
	_build_caution_schedule_matrix,
	_enumerate_unconditioned_subject_strategies,
	_precompute_subject_inputs,
	_resolve_probability_by_source,
	argmax_strategy,
)
from source_code.simulation.race_model import _driver_pit_loss, simulate_car_race
from source_code.simulation.race_model_vectorized import evaluate_strategy_batch, rank_and_score_batch
from source_code.simulation.race_state import load_race_state


DEG_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_full_peryear_stats.csv"
OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step5b_degenerate_stint_diagnosis_output.txt"

HUNGARY_2022 = {"event_name": "Hungarian Grand Prix", "year": 2022, "round": 13}
ITALY_2024 = {"event_name": "Italian Grand Prix", "year": 2024, "round": 16}

PROB_SOURCE = "lap_level"
INFORMATION_SET = "unconditioned"
LAMBDA_VALUE = 1.0
N_ITERATIONS = 10_000
SEED = 42
EPSILON = 0.01


def emit(text: str = "") -> None:
	print(text, flush=True)


def triangular_number(n: int) -> int:
	n = int(n)
	return n * (n + 1) // 2


def _to_bool(series: pd.Series) -> pd.Series:
	if pd.api.types.is_bool_dtype(series):
		return series.fillna(False)
	return series.fillna(False).astype(bool)


def strategy_to_jsonable(strategy: Mapping[str, object]) -> Dict[str, object]:
	return {
		"baseline_pace_s": float(strategy["baseline_pace_s"]),
		"event_name": str(strategy["event_name"]),
		"round": int(strategy["round"]),
		"starting_compound": str(strategy["starting_compound"]),
		"stops": [[int(lap), str(compound)] for lap, compound in strategy.get("stops", [])],
		"year": int(strategy["year"]),
	}


def compound_sequence_and_stint_lengths(strategy: Mapping[str, object], race_length: int) -> Tuple[List[str], List[int]]:
	stops = [(int(lap), str(compound)) for lap, compound in strategy.get("stops", [])]
	compounds = [str(strategy["starting_compound"])]
	stint_lengths: List[int] = []
	previous_stop = 0
	for pit_lap, compound in stops:
		stint_lengths.append(int(pit_lap) - previous_stop)
		compounds.append(compound)
		previous_stop = int(pit_lap)
	stint_lengths.append(int(race_length) - previous_stop)
	return compounds, stint_lengths


def green_degradation_cost(strategy: Mapping[str, object], race_length: int, rate_by_compound: Mapping[str, float]) -> Dict[str, object]:
	compounds, stint_lengths = compound_sequence_and_stint_lengths(strategy, race_length)
	rows = []
	total = 0.0
	for compound, stint_length in zip(compounds, stint_lengths):
		rate = float(rate_by_compound[compound])
		tri = triangular_number(stint_length)
		cost = rate * tri
		rows.append(
			{
				"compound": compound,
				"stint_length": int(stint_length),
				"triangular_number": int(tri),
				"deg_rate_final_assigned": rate,
				"degradation_cost_s": cost,
			}
		)
		total += cost
	return {"rows": rows, "total_degradation_cost_s": float(total)}


def numeric_single_stint_confirmation(compound: str, stint_length: int, rate: float) -> Dict[str, float]:
	strategy = {
		"starting_compound": str(compound),
		"baseline_pace_s": 0.0,
		"stops": [],
	}
	result = simulate_car_race(
		strategy=strategy,
		caution_schedule={lap: False for lap in range(1, int(stint_length) + 1)},
		degradation_stats=None,
		race_length=int(stint_length),
		lambda_=1.0,
		pit_loss_s=0.0,
		caution_pit_loss_s=0.0,
		caution_pace_ratio=CAUTION_PACE_RATIO,
		degradation_lookup={(str(compound),): float(rate)},
		collect_trace=False,
	)
	expected = float(rate) * triangular_number(int(stint_length))
	return {
		"simulated_total_s": float(result["race_time_s"]),
		"expected_total_s": float(expected),
		"abs_diff_s": abs(float(result["race_time_s"]) - float(expected)),
	}


def get_rate_by_compound(deg_df: pd.DataFrame, year: int, event_name: str) -> Dict[str, float]:
	subset = deg_df[
		(deg_df["Year"] == int(year))
		& (deg_df["EventName"] == str(event_name))
		& (deg_df["TyreCompound"].isin(["SOFT", "MEDIUM", "HARD"]))
	].copy()
	lookup = {}
	for _, row in subset.iterrows():
		value = row["deg_rate_final_assigned"]
		if pd.notna(value):
			lookup[str(row["TyreCompound"])] = float(value)
	return lookup


def prepare_search(race_ref: Mapping[str, object], prob_source: str) -> Dict[str, object]:
	state = load_race_state(int(race_ref["year"]), int(race_ref["round"]))
	laps = state["laps"].copy()
	laps["sc_active"] = _to_bool(laps["sc_active"])
	laps["vsc_active"] = _to_bool(laps["vsc_active"])

	subject_driver = select_midfield_subject_driver(laps)
	actual_strategies = build_actual_strategies(laps)
	subject_actual = dict(actual_strategies[subject_driver])
	race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

	prob_by_lap = _resolve_probability_by_source(state, prob_source)
	caution_schedules = _build_caution_schedule_matrix(prob_by_lap, N_ITERATIONS, np.random.default_rng(SEED))
	strategies = _enumerate_unconditioned_subject_strategies(race_length, subject_actual)
	precomputed = _precompute_subject_inputs(
		race_state=state,
		subject_driver=subject_driver,
		caution_schedules=caution_schedules,
		lambda_=LAMBDA_VALUE,
	)

	return {
		"race_ref": dict(race_ref),
		"race_state": state,
		"laps": laps,
		"subject_driver": subject_driver,
		"subject_actual": subject_actual,
		"actual_strategies": actual_strategies,
		"race_length": race_length,
		"caution_schedules": caution_schedules,
		"strategies": strategies,
		"precomputed": precomputed,
	}


def score_strategies(search: Mapping[str, object], strategies: Sequence[Mapping[str, object]]) -> Dict[str, np.ndarray]:
	mean_points = np.zeros(len(strategies), dtype=np.float64)
	std_points = np.zeros(len(strategies), dtype=np.float64)
	mean_rank = np.zeros(len(strategies), dtype=np.float64)

	total_batches = (len(strategies) + ARGMAX_BATCH_SIZE - 1) // ARGMAX_BATCH_SIZE

	for start in range(0, len(strategies), ARGMAX_BATCH_SIZE):
		end = min(start + ARGMAX_BATCH_SIZE, len(strategies))
		batch = list(strategies[start:end])
		batch_idx = (start // ARGMAX_BATCH_SIZE) + 1
		subject_times = evaluate_strategy_batch(
			subject_strategies=batch,
			caution_schedules=search["caution_schedules"],
			degradation_lookup=search["precomputed"]["degradation_lookup"],
			lambda_=LAMBDA_VALUE,
			baseline_pace_s=float(search["subject_actual"]["baseline_pace_s"]),
			pit_loss_s=float(search["precomputed"]["pit_loss_s"]),
			caution_pit_loss_s=CAUTION_PIT_LOSS_S,
			caution_pace_ratio=CAUTION_PACE_RATIO,
		)
		scored = rank_and_score_batch(subject_times, search["precomputed"]["rival_times"])
		mean_points[start:end] = np.asarray(scored["mean_points"], dtype=np.float64)
		std_points[start:end] = np.asarray(scored["std_points"], dtype=np.float64)
		mean_rank[start:end] = np.asarray(scored["mean_rank"], dtype=np.float64)
		if batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0:
			emit(
				f"batch_progress race={search['race_ref']['event_name']} "
				f"batch={batch_idx}/{total_batches} scored_strategies={end}/{len(strategies)}"
			)

	return {
		"mean_points": mean_points,
		"std_points": std_points,
		"mean_rank": mean_rank,
	}


def evaluate_single_strategy(search: Mapping[str, object], strategy: Mapping[str, object]) -> Dict[str, object]:
	subject_times = evaluate_strategy_batch(
		subject_strategies=[dict(strategy)],
		caution_schedules=search["caution_schedules"],
		degradation_lookup=search["precomputed"]["degradation_lookup"],
		lambda_=LAMBDA_VALUE,
		baseline_pace_s=float(search["subject_actual"]["baseline_pace_s"]),
		pit_loss_s=float(search["precomputed"]["pit_loss_s"]),
		caution_pit_loss_s=CAUTION_PIT_LOSS_S,
		caution_pace_ratio=CAUTION_PACE_RATIO,
	)
	scored = rank_and_score_batch(subject_times, search["precomputed"]["rival_times"])
	return {
		"subject_times": np.asarray(subject_times[0], dtype=np.float64),
		"rank_matrix": np.asarray(scored["rank_matrix"][0], dtype=np.int16),
		"points_matrix": np.asarray(scored["points_matrix"][0], dtype=np.float64),
		"mean_points": float(scored["mean_points"][0]),
		"std_points": float(scored["std_points"][0]),
		"mean_rank": float(scored["mean_rank"][0]),
	}


def locked_race_asymmetry_table(deg_df: pd.DataFrame) -> pd.DataFrame:
	lock_df = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
	pivot = (
		deg_df[
			deg_df["TyreCompound"].isin(["SOFT", "MEDIUM", "HARD"])
			& deg_df["deg_rate_final_assigned"].notna()
		]
		.pivot_table(
			index=["EventName", "Year"],
			columns="TyreCompound",
			values="deg_rate_final_assigned",
			aggfunc="first",
		)
		.reset_index()
	)
	merged = lock_df.merge(
		pivot,
		left_on=["event_name", "year"],
		right_on=["EventName", "Year"],
		how="left",
	)
	merged["hard_minus_soft"] = merged["HARD"] - merged["SOFT"]
	merged["hard_minus_medium"] = merged["HARD"] - merged["MEDIUM"]
	merged["abs_asymmetry_magnitude"] = merged[["hard_minus_soft", "hard_minus_medium"]].abs().max(axis=1, skipna=True)
	merged = merged.sort_values(["abs_asymmetry_magnitude", "event_name", "year"], ascending=[False, True, True])
	return merged[[
		"archetype",
		"event_name",
		"year",
		"round",
		"HARD",
		"MEDIUM",
		"SOFT",
		"hard_minus_soft",
		"hard_minus_medium",
		"abs_asymmetry_magnitude",
	]]


def main() -> None:
	deg_df = pd.read_csv(DEG_STATS_PATH)

	emit("=" * 100)
	emit("Step 5b setup")
	emit("=" * 100)
	emit(f"output_path={OUTPUT_PATH}")
	emit(f"prob_source={PROB_SOURCE}")
	emit(f"information_set={INFORMATION_SET}")
	emit(f"n_iterations={N_ITERATIONS}")
	emit(f"seed={SEED}")
	emit("progress=loading Hungary 2022 race state and reproducing argmax candidate")

	hungary_search = prepare_search(HUNGARY_2022, PROB_SOURCE)
	hungary_subject_laps = hungary_search["laps"][hungary_search["laps"]["Driver"].astype(str) == hungary_search["subject_driver"]].copy()
	hungary_pit_loss = float(_driver_pit_loss(hungary_subject_laps))
	hungary_rates = get_rate_by_compound(deg_df, HUNGARY_2022["year"], HUNGARY_2022["event_name"])
	hungary_actual = dict(hungary_search["subject_actual"])
	emit("progress=running Hungary 2022 argmax reproduction")
	hungary_best = argmax_strategy(
		race_state=hungary_search["race_state"],
		subject_driver=hungary_search["subject_driver"],
		prob_source=PROB_SOURCE,
		information_set=INFORMATION_SET,
		lambda_=LAMBDA_VALUE,
		n_iterations=N_ITERATIONS,
		rng=np.random.default_rng(SEED),
		epsilon=EPSILON,
	)
	hungary_best_strategy = dict(hungary_best["best_strategy"])

	emit("=" * 100)
	emit("Step 1 - Quadratic stint-length degradation cost confirmation")
	emit("=" * 100)
	emit("race_model_formula=green lap_time_s = baseline_pace_s + degradation_rate * effective_age; effective_age starts at 1.0, increments by 1.0 after each green lap, and resets to 1.0 only on pit laps")
	emit("closed_form_green_stint_cost=degradation_rate * sum_{k=1..n} k = degradation_rate * n(n+1)/2")
	emit(f"hungary_subject_driver={hungary_search['subject_driver']}")
	emit(f"hungary_subject_green_pit_loss_s={hungary_pit_loss:.6f}")
	emit(f"hungary_deg_rate_HARD={hungary_rates['HARD']:.15f}")
	emit(f"hungary_deg_rate_MEDIUM={hungary_rates['MEDIUM']:.15f}")
	emit(f"hungary_deg_rate_SOFT={hungary_rates['SOFT']:.15f}")

	for compound, stint_length in [("SOFT", 1), ("SOFT", 15), ("HARD", 69), ("MEDIUM", 28), ("HARD", 27)]:
		confirmation = numeric_single_stint_confirmation(compound, stint_length, hungary_rates[compound])
		emit(
			f"single_stint_numeric_check compound={compound} stint_length={stint_length} "
			f"expected_s={confirmation['expected_total_s']:.12f} simulated_s={confirmation['simulated_total_s']:.12f} "
			f"abs_diff_s={confirmation['abs_diff_s']:.12e}"
		)

	hungary_candidate = dict(hungary_best_strategy)
	actual_cost = green_degradation_cost(hungary_actual, hungary_search["race_length"], hungary_rates)
	candidate_cost = green_degradation_cost(hungary_candidate, hungary_search["race_length"], hungary_rates)
	actual_total_offset = actual_cost["total_degradation_cost_s"] + hungary_pit_loss * len(hungary_actual.get("stops", []))
	candidate_total_offset = candidate_cost["total_degradation_cost_s"] + hungary_pit_loss * len(hungary_candidate.get("stops", []))

	emit("hungary_actual_strategy=" + json.dumps(strategy_to_jsonable(hungary_actual), sort_keys=True))
	emit("hungary_candidate_argmax_strategy=" + json.dumps(strategy_to_jsonable(hungary_candidate), sort_keys=True))
	emit("hungary_actual_stint_breakdown_begin")
	for row in actual_cost["rows"]:
		emit(json.dumps(row, sort_keys=True))
	emit("hungary_actual_stint_breakdown_end")
	emit("hungary_candidate_stint_breakdown_begin")
	for row in candidate_cost["rows"]:
		emit(json.dumps(row, sort_keys=True))
	emit("hungary_candidate_stint_breakdown_end")
	emit(f"hungary_actual_total_degradation_cost_s={actual_cost['total_degradation_cost_s']:.6f}")
	emit(f"hungary_candidate_total_degradation_cost_s={candidate_cost['total_degradation_cost_s']:.6f}")
	emit(f"hungary_actual_total_subject_offset_vs_baseline_s={actual_total_offset:.6f}")
	emit(f"hungary_candidate_total_subject_offset_vs_baseline_s={candidate_total_offset:.6f}")
	emit(f"hungary_candidate_advantage_vs_actual_s={actual_total_offset - candidate_total_offset:.6f}")
	emit("interpretation=Under the current formula, the 1-lap SOFT then long HARD plan converts the strong positive SOFT age slope into a single lap while exploiting the slightly negative HARD slope over 69 laps, and it also saves one full green-flag pit loss relative to the actual 2-stop strategy.")

	emit("")
	emit("=" * 100)
	emit("Step 2 - Hungary full-search first-stint distribution and top-100 concentration")
	emit("=" * 100)
	emit("progress=scoring full Hungary 2022 unconditioned search")
	hungary_scores = score_strategies(hungary_search, hungary_search["strategies"])
	first_stint_lengths = np.array([compound_sequence_and_stint_lengths(strategy, hungary_search["race_length"])[1][0] for strategy in hungary_search["strategies"]], dtype=np.int16)
	distribution = (
		pd.Series(first_stint_lengths)
		.value_counts()
		.sort_index()
	)
	emit(f"hungary_strategy_count={len(hungary_search['strategies'])}")
	emit("hungary_first_stint_length_distribution_begin")
	for stint_length, count in distribution.items():
		emit(f"first_stint_length={int(stint_length)} count={int(count)} share={float(count / len(first_stint_lengths)):.6f}")
	emit("hungary_first_stint_length_distribution_end")

	top100_idx = np.argsort(-hungary_scores["mean_points"])[:100]
	top100_short_count = int((first_stint_lengths[top100_idx] <= 3).sum())
	emit(f"hungary_top100_short_first_stint_leq3_count={top100_short_count}")
	emit(f"hungary_top100_short_first_stint_leq3_fraction={top100_short_count / 100.0:.6f}")
	emit("hungary_top10_strategies_begin")
	for idx in top100_idx[:10]:
		row = {
			"strategy_index": int(idx),
			"first_stint_length": int(first_stint_lengths[idx]),
			"mean_points": float(hungary_scores["mean_points"][idx]),
			"std_points": float(hungary_scores["std_points"][idx]),
			"strategy": strategy_to_jsonable(hungary_search["strategies"][idx]),
		}
		emit(json.dumps(row, sort_keys=True))
	emit("hungary_top10_strategies_end")

	emit("")
	emit("=" * 100)
	emit("Step 3 - Italian GP 2024 exact-25.0 diagnosis")
	emit("=" * 100)
	emit("progress=loading Italian GP 2024 race state and reproducing argmax candidate")
	italy_search = prepare_search(ITALY_2024, PROB_SOURCE)
	emit("progress=running Italian GP 2024 argmax reproduction")
	italy_best = argmax_strategy(
		race_state=italy_search["race_state"],
		subject_driver=italy_search["subject_driver"],
		prob_source=PROB_SOURCE,
		information_set=INFORMATION_SET,
		lambda_=LAMBDA_VALUE,
		n_iterations=N_ITERATIONS,
		rng=np.random.default_rng(SEED),
		epsilon=EPSILON,
	)
	italy_best_strategy = dict(italy_best["best_strategy"])
	italy_single = evaluate_single_strategy(italy_search, italy_best_strategy)
	italy_rival_times = np.asarray(italy_search["precomputed"]["rival_times"], dtype=np.float64)
	italy_subject_times = np.asarray(italy_single["subject_times"], dtype=np.float64)
	italy_subject_mean_time = float(italy_subject_times.mean())
	italy_win_rate = float(np.mean(italy_single["rank_matrix"] == 1))
	italy_all_first = bool(np.all(italy_single["rank_matrix"] == 1))
	italy_compounds, italy_stint_lengths = compound_sequence_and_stint_lengths(italy_best_strategy, italy_search["race_length"])
	italy_rivals = sorted([driver for driver in italy_search["actual_strategies"] if driver != italy_search["subject_driver"]])
	italy_rival_mean_rows = []
	for rival_idx, rival_driver in enumerate(italy_rivals):
		rival_mean_time = float(italy_rival_times[rival_idx].mean())
		italy_rival_mean_rows.append(
			{
				"rival_driver": rival_driver,
				"rival_mean_race_time_s": rival_mean_time,
				"subject_minus_rival_mean_time_s": italy_subject_mean_time - rival_mean_time,
			}
		)
	italy_rival_mean_rows.sort(key=lambda row: row["rival_mean_race_time_s"])
	closest_rival_by_mean = min(italy_rival_mean_rows, key=lambda row: row["rival_mean_race_time_s"])
	closest_margin_by_iteration = np.min(italy_rival_times - italy_subject_times[None, :], axis=0)

	emit(f"italy_subject_driver={italy_search['subject_driver']}")
	emit("italy_best_strategy=" + json.dumps(strategy_to_jsonable(italy_best_strategy), sort_keys=True))
	emit(f"italy_best_mean_points={italy_best['best_mean_points']:.6f}")
	emit(f"italy_best_mean_rank={italy_single['mean_rank']:.6f}")
	emit(f"italy_exact_win_rate={italy_win_rate:.6f}")
	emit(f"italy_all_iterations_rank1={italy_all_first}")
	emit(f"italy_stint_compounds={json.dumps(italy_compounds)}")
	emit(f"italy_stint_lengths={json.dumps([int(x) for x in italy_stint_lengths])}")
	emit(f"italy_first_stint_length={int(italy_stint_lengths[0])}")
	emit(f"italy_first_stint_leq3={bool(int(italy_stint_lengths[0]) <= 3)}")
	emit(f"italy_subject_mean_race_time_s={italy_subject_mean_time:.6f}")
	emit("italy_rival_mean_time_comparison_begin")
	for row in italy_rival_mean_rows:
		emit(json.dumps(row, sort_keys=True))
	emit("italy_rival_mean_time_comparison_end")
	emit("italy_closest_rival_by_mean=" + json.dumps(closest_rival_by_mean, sort_keys=True))
	if italy_all_first:
		emit(f"italy_closest_rival_min_margin_over_all_iterations_s={float(np.min(closest_margin_by_iteration)):.6f}")
		emit(f"italy_closest_rival_mean_margin_s={float(closest_rival_by_mean['rival_mean_race_time_s'] - italy_subject_mean_time):.6f}")
		emit("italy_interpretation=The exact 25.0 mean comes from a 100% simulated win rate, not from a scoring bug in the points mapping itself. The remaining question is whether that universal dominance is physically plausible under the current degradation model or an exploit of missing minimum stint length.")
	else:
		emit("italy_interpretation=The 25.0 mean is not supported by a 100% win rate; investigate scoring logic immediately.")

	emit("")
	emit("=" * 100)
	emit("Step 4 - Locked-race compound asymmetry ranking")
	emit("=" * 100)
	asymmetry = locked_race_asymmetry_table(deg_df)
	emit("locked_race_asymmetry_begin")
	for row in asymmetry.to_dict(orient="records"):
		emit(json.dumps({k: (None if pd.isna(v) else v) for k, v in row.items()}, sort_keys=True))
	emit("locked_race_asymmetry_end")

	hungary_asym = asymmetry[
		(asymmetry["event_name"] == HUNGARY_2022["event_name"]) & (asymmetry["year"] == HUNGARY_2022["year"])
	].iloc[0]
	italy_asym = asymmetry[
		(asymmetry["event_name"] == ITALY_2024["event_name"]) & (asymmetry["year"] == ITALY_2024["year"])
	].iloc[0]
	emit(f"hungary_2022_abs_asymmetry_magnitude={float(hungary_asym['abs_asymmetry_magnitude']):.6f}")
	emit(f"italy_2024_abs_asymmetry_magnitude={float(italy_asym['abs_asymmetry_magnitude']):.6f}")
	emit(f"italy_2024_hard_minus_soft={None if pd.isna(italy_asym['hard_minus_soft']) else float(italy_asym['hard_minus_soft'])}")
	emit(f"italy_2024_hard_minus_medium={None if pd.isna(italy_asym['hard_minus_medium']) else float(italy_asym['hard_minus_medium'])}")
	emit(f"hungary_2022_hard_minus_soft={None if pd.isna(hungary_asym['hard_minus_soft']) else float(hungary_asym['hard_minus_soft'])}")
	emit(f"hungary_2022_hard_minus_medium={None if pd.isna(hungary_asym['hard_minus_medium']) else float(hungary_asym['hard_minus_medium'])}")
	if float(italy_asym["abs_asymmetry_magnitude"]) > float(hungary_asym["abs_asymmetry_magnitude"]):
		emit("asymmetry_comparison_statement=Italian GP 2024 is more asymmetric than Hungary 2022 on the available locked-race gap measure.")
	else:
		emit("asymmetry_comparison_statement=Italian GP 2024 is not more asymmetric than Hungary 2022 on the available locked-race gap measure; the exact-25 outcome therefore also depends on race-specific baseline pace/rival context, not just a uniquely extreme compound gap.")

	emit("")
	emit("=" * 100)
	emit("Step 5 - Recommended minimum-stint rule (report only)")
	emit("=" * 100)
	emit("recommended_minimum_stint_length_laps=16")
	emit("recommendation_reasoning=With green-flag pit losses on the order of ~20s and even the most aggressive locked-race HARD-vs-SOFT/MEDIUM slope gaps staying around a few tenths of a second per unit effective age, sub-16-lap stints cannot rationally repay a pit stop through degradation relief alone under this model. A 16-lap floor is therefore a physically grounded anti-exploit rule that is independent of contaminated historical stint-length percentiles and directly targets the pathological 1-lap/4-lap search outcomes exposed here.")
	emit("recommendation_scope=Do not trust any B/(a)/(b) benchmark that consumes optimiser argmax outputs until strategy enumeration enforces a non-trivial minimum stint length.")

	emit("")
	emit(f"tee_output_target={OUTPUT_PATH}")


if __name__ == "__main__":
	main()