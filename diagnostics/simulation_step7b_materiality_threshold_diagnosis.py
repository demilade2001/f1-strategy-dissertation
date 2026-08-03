from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation import config as sim_config  # noqa: E402
from source_code.simulation.optimiser import argmax_strategy  # noqa: E402
from source_code.simulation.race_state import load_race_state  # noqa: E402


STEP7_OUTPUT = ROOT / "data" / "diagnostics" / "simulation_step7_bias_decomposition_output.txt"
OUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step7b_materiality_threshold_diagnosis_output.txt"
DEG_CORRECTED_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_v3_full_peryear_stats.csv"

N300 = 300
N10000 = 10000

TARGET_RACES = [
    {"year": 2024, "round": 1, "event_name": "Bahrain Grand Prix", "subject_driver": "NOR"},
    {"year": 2022, "round": 13, "event_name": "Hungarian Grand Prix", "subject_driver": "NOR"},
    {"year": 2024, "round": 6, "event_name": "Miami Grand Prix", "subject_driver": "NOR"},
]


def emit(out_handle, text: str = "") -> None:
    print(text, flush=True)
    out_handle.write(text + "\n")
    out_handle.flush()


def qstats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "median": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "max": math.nan,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _load_race_state_with_corrected_deg(race: Mapping[str, Any]) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    if DEG_CORRECTED_PATH.exists():
        deg = pd.read_csv(DEG_CORRECTED_PATH)
        event_name = str(state["laps"]["EventName"].dropna().iloc[0])
        state["deg_stats"] = deg[(deg["Year"] == int(race["year"])) & (deg["EventName"] == event_name)].copy()
    return state


def _compute_b_mean_points(race: Mapping[str, Any], n_iterations: int, seed: int) -> float:
    state = _load_race_state_with_corrected_deg(race)
    result = argmax_strategy(
        race_state=state,
        subject_driver=str(race["subject_driver"]),
        prob_source="lap_level",
        information_set="unconditioned",
        lambda_=1.0,
        n_iterations=n_iterations,
        rng=np.random.default_rng(seed),
        epsilon=0.01,
    )
    return float(result["best_mean_points"])


def _parse_step7_output(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Required Step 7 output not found: {path}")

    anchoring_rows: List[Dict[str, Any]] = []
    sc_rows: List[Dict[str, Any]] = []
    section = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line == "anchoring_rows_begin":
                section = "anchoring"
                continue
            if line == "anchoring_rows_end":
                section = None
                continue
            if line == "sc_rows_begin":
                section = "sc"
                continue
            if line == "sc_rows_end":
                section = None
                continue

            if section not in {"anchoring", "sc"}:
                continue
            if not line.startswith("{"):
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if section == "anchoring" and bool(row.get("computable", False)):
                anchoring_rows.append(row)
            if section == "sc" and bool(row.get("computable", False)):
                sc_rows.append(row)

    return anchoring_rows, sc_rows


def _abs_gap(rows: List[Mapping[str, Any]], b_key: str, r_key: str) -> np.ndarray:
    vals = [abs(float(r[b_key]) - float(r[r_key])) for r in rows if pd.notna(r.get(b_key)) and pd.notna(r.get(r_key))]
    return np.asarray(vals, dtype=np.float64)


def _strict_unidentifiable_count(rows: List[Mapping[str, Any]]) -> int:
    c = 0
    for r in rows:
        rb = r.get("r_b")
        unidentifiable = bool(r.get("unidentifiable", False))
        if unidentifiable:
            c += 1
            continue
        try:
            if pd.isna(rb):
                c += 1
        except Exception:
            pass
    return c


def _select_threshold(combined_abs_gap: np.ndarray, noise_floor: float) -> Tuple[float, int, float]:
    required_min = noise_floor * 1.5
    candidate_percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 50]
    selected_p = candidate_percentiles[-1]
    selected_v = float(np.quantile(combined_abs_gap, selected_p / 100.0))
    for p in candidate_percentiles:
        v = float(np.quantile(combined_abs_gap, p / 100.0))
        if v >= required_min:
            selected_p = p
            selected_v = v
            break
    return selected_v, selected_p, required_min


def _materiality_impact(rows: List[Mapping[str, Any]], b_key: str, r_key: str, threshold: float) -> Dict[str, Any]:
    abs_gap = np.asarray([abs(float(r[b_key]) - float(r[r_key])) for r in rows], dtype=np.float64)
    strict_flags = np.asarray(
        [bool(r.get("unidentifiable", False)) or pd.isna(r.get("r_b")) for r in rows],
        dtype=bool,
    )
    materiality_flags = abs_gap < threshold
    materiality_only = materiality_flags & (~strict_flags)
    identifiable = ~(strict_flags | materiality_flags)

    rb_all = np.asarray([float(r.get("r_b", np.nan)) for r in rows], dtype=np.float64)
    rb_ident = rb_all[identifiable]
    rb_ident = rb_ident[~np.isnan(rb_ident)]

    outlier_before = int(np.sum(np.abs(rb_all[~np.isnan(rb_all)]) > 5.0))
    outlier_after = int(np.sum(np.abs(rb_ident) > 5.0))

    return {
        "total_rows": int(len(rows)),
        "strict_unidentifiable_count": int(np.sum(strict_flags)),
        "materiality_flag_count": int(np.sum(materiality_flags)),
        "materiality_only_count": int(np.sum(materiality_only)),
        "combined_unidentifiable_count": int(np.sum(strict_flags | materiality_flags)),
        "materially_identifiable_count": int(np.sum(identifiable)),
        "abs_gap_distribution": qstats(abs_gap.tolist()),
        "r_b_identifiable_distribution": qstats(rb_ident.tolist()),
        "outlier_abs_rb_gt_5_before": outlier_before,
        "outlier_abs_rb_gt_5_after": outlier_after,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, "Step 7b - Materiality threshold diagnosis (report only, no formula change)")
        emit(out, "=" * 100)

        emit(out, "")
        emit(out, "Step 1 - Repeatability check on B")
        emit(out, f"n_iterations_300={N300}")
        emit(out, f"n_iterations_10000={N10000}")

        seeds_300 = [sim_config.RNG_SEED + 11, sim_config.RNG_SEED + 211, sim_config.RNG_SEED + 1211]
        seed_10000 = sim_config.RNG_SEED + 10001

        repeatability_rows: List[Dict[str, Any]] = []
        spreads = []
        for race in TARGET_RACES:
            vals_300 = [_compute_b_mean_points(race, N300, s) for s in seeds_300]
            val_10000 = _compute_b_mean_points(race, N10000, seed_10000)
            spread_300 = float(max(vals_300) - min(vals_300))
            spreads.append(spread_300)
            row = {
                "race": race,
                "B_mean_points_n300_run1": vals_300[0],
                "B_mean_points_n300_run2": vals_300[1],
                "B_mean_points_n300_run3": vals_300[2],
                "B_mean_points_n300_spread": spread_300,
                "B_mean_points_n10000_reference": val_10000,
            }
            repeatability_rows.append(row)
            emit(out, json.dumps(row, sort_keys=True, default=str))

        noise_floor = float(max(spreads)) if spreads else math.nan
        emit(out, f"n300_noise_floor_max_spread={noise_floor}")
        emit(out, "comparison_target_gap_band_points=[0.07, 2.0]")
        if noise_floor >= 0.07:
            emit(out, "n300_noise_statement=At least part of the observed B-R gap band (0.07 to ~2 points) is within MC spread at n=300.")
        else:
            emit(out, "n300_noise_statement=MC spread at n=300 is below the lower edge of the observed 0.07 to ~2 point gap band.")

        emit(out, "")
        emit(out, "Step 2 - Distribution of |B-R| across pilot")
        anch_rows, sc_rows = _parse_step7_output(STEP7_OUTPUT)
        emit(out, f"anchoring_rows_parsed={len(anch_rows)}")
        emit(out, f"sc_rows_parsed={len(sc_rows)}")

        anch_abs = _abs_gap(anch_rows, "B_mean_points", "R_anchoring_mean_points")
        sc_abs = _abs_gap(sc_rows, "B_lap_level_mean_points", "R_sc_static_prior_mean_points")

        anch_dist = qstats(anch_abs.tolist())
        sc_dist = qstats(sc_abs.tolist())
        emit(out, "anchoring_abs_B_minus_R_distribution=" + json.dumps(anch_dist, sort_keys=True, default=str))
        emit(out, "sc_abs_B_minus_R_distribution=" + json.dumps(sc_dist, sort_keys=True, default=str))

        emit(out, "")
        emit(out, "Step 3 - Propose percentile-derived materiality threshold")
        combined_abs = np.concatenate([anch_abs, sc_abs])
        threshold, percentile, required_min = _select_threshold(combined_abs, noise_floor)
        emit(out, f"materiality_threshold_percentile=p{percentile}")
        emit(out, f"materiality_threshold_abs_B_minus_R={threshold}")
        emit(out, f"required_min_from_noise_floor(1.5x)={required_min}")
        emit(
            out,
            "materiality_justification="
            + "Selected the lowest percentile whose abs(B-R) value is at least 1.5x the measured n=300 noise floor.",
        )

        emit(out, "")
        emit(out, "Step 4 - Materiality impact and post-filter r_b distribution")
        anch_impact = _materiality_impact(anch_rows, "B_mean_points", "R_anchoring_mean_points", threshold)
        sc_impact = _materiality_impact(sc_rows, "B_lap_level_mean_points", "R_sc_static_prior_mean_points", threshold)
        emit(out, "anchoring_materiality_impact=" + json.dumps(anch_impact, sort_keys=True, default=str))
        emit(out, "sc_materiality_impact=" + json.dumps(sc_impact, sort_keys=True, default=str))

        total_outliers_before = anch_impact["outlier_abs_rb_gt_5_before"] + sc_impact["outlier_abs_rb_gt_5_before"]
        total_outliers_after = anch_impact["outlier_abs_rb_gt_5_after"] + sc_impact["outlier_abs_rb_gt_5_after"]
        if total_outliers_after == 0:
            emit(out, "outlier_statement=All |r_b|>5 outliers are removed after materiality filtering.")
        else:
            emit(out, "outlier_statement=|r_b|>5 outliers persist after materiality filtering; this indicates a deeper modelling issue.")
        emit(out, f"combined_outliers_abs_rb_gt_5_before={total_outliers_before}")
        emit(out, f"combined_outliers_abs_rb_gt_5_after={total_outliers_after}")

        emit(out, "")
        emit(out, "Step 5 - Recommendation (do not implement)")
        if noise_floor >= 0.2:
            rec = 10000
            rationale = "n=300 spread is large enough to contaminate low-gap rows; keep validated standard for reference computations."
        elif noise_floor >= 0.07:
            rec = 5000
            rationale = "n=300 spread overlaps the lower gap band; raise iterations substantially for stable denominators."
        else:
            rec = 2000
            rationale = "n=300 spread is below the lower gap band, but a moderate increase still improves denominator stability."

        recommendation = {
            "recommended_n_iterations_for_B_and_R_reference_computations": rec,
            "rationale": rationale,
            "note": "Use the existing vectorized/single-strategy evaluation path where available to keep runtime tractable.",
        }
        emit(out, "recommendation=" + json.dumps(recommendation, sort_keys=True, default=str))

    print(f"output_file={OUT_PATH}")
    print(f"output_file_bytes={OUT_PATH.stat().st_size if OUT_PATH.exists() else 0}")


if __name__ == "__main__":
    main()