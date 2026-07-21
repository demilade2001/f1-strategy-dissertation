from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.monte_carlo import run_monte_carlo
from src.simulation.race_model import simulate_car_race, simulate_race_with_driver_laps
from src.simulation.race_state import load_race_state

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step4f_vectorization_gap_diagnosis_output.txt"
STEP2H_SCRIPT_PATH = ROOT / "diagnostics" / "simulation_step2h_locked_sample_tail_and_runtime.py"
STEP2H_OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step2h_locked_sample_tail_and_runtime_output.txt"


def load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_key_value_from_output(path: Path, key: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected diagnostics output file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise KeyError(f"Key {key} not found in {path}")


def main() -> None:
    lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 100)
    emit("Step 1 - Core per-iteration loop structure in race_model.py and monte_carlo.py")
    emit("=" * 100)

    emit("simulate_car_race source:")
    emit(inspect.getsource(simulate_car_race).rstrip())
    emit("")

    emit("simulate_race_with_driver_laps source:")
    emit(inspect.getsource(simulate_race_with_driver_laps).rstrip())
    emit("")

    emit("run_monte_carlo source:")
    emit(inspect.getsource(run_monte_carlo).rstrip())
    emit("")

    state = load_race_state(2022, 13)
    laps = state["laps"]
    driver_count = int(laps["Driver"].astype(str).nunique())
    race_length = int(laps["LapNumber"].max())
    n_iterations = 10_000
    lap_updates = n_iterations * driver_count * race_length

    emit("Loop classification:")
    emit("- Python loop over Monte Carlo iterations in run_monte_carlo: for _ in range(n_iterations)")
    emit("- Python loop over drivers in simulate_race_from_prepared (called by run_monte_carlo)")
    emit("- Python loop over laps in simulate_car_race: for lap_number in range(1, race_length + 1)")
    emit("- No strategy-batch vectorization in the package path; NumPy usage is limited to simple array summaries and RNG.")
    emit(f"Approximate nesting for one run_monte_carlo call on Hungary 2022: {n_iterations} iterations x {driver_count} drivers x {race_length} laps")
    emit(f"Approximate lap-level updates per strategy evaluation: {lap_updates:,}")

    emit("")
    emit("=" * 100)
    emit("Step 2 - Vectorization approach used in step 2h benchmark script")
    emit("=" * 100)

    step2h_module = load_module_from_path("simulation_step2h_module", STEP2H_SCRIPT_PATH)

    emit("run_full_vectorized_benchmark source:")
    emit(inspect.getsource(step2h_module.run_full_vectorized_benchmark).rstrip())
    emit("")

    emit("enumerate_strategy_features source:")
    emit(inspect.getsource(step2h_module.enumerate_strategy_features).rstrip())
    emit("")

    bench_race = extract_key_value_from_output(STEP2H_OUTPUT_PATH, "benchmark_race")
    bench_strategy_count = int(extract_key_value_from_output(STEP2H_OUTPUT_PATH, "benchmark_strategy_count"))
    bench_schedule_count = int(extract_key_value_from_output(STEP2H_OUTPUT_PATH, "benchmark_schedule_count"))
    bench_seconds = float(extract_key_value_from_output(STEP2H_OUTPUT_PATH, "benchmark_wall_clock_seconds"))
    bench_race_length = int(extract_key_value_from_output(STEP2H_OUTPUT_PATH, "benchmark_race_length"))

    emit("Vectorization summary for step 2h:")
    emit(f"- Strategies dimension N = {bench_strategy_count}")
    emit(f"- Schedules dimension S = {bench_schedule_count}")
    emit(f"- Laps dimension L = {bench_race_length}")
    emit("- Key arrays: caution_schedules shape (S, L), stop_mask shape (N, L), caution_hits shape (N, B) per batch")
    emit("- Ranking and points computed with array argsort/rank operations across N strategies for each schedule batch")
    emit(f"- Reported runtime: {bench_seconds:.6f} seconds for N={bench_strategy_count}, S={bench_schedule_count}")
    emit("- Physics level is simplified vs race_model: base_time uses coarse formula with stop counts, compound counts, fixed caution benefit, and Gaussian noise")
    emit("- It does not execute per-driver, per-lap degradation aging, lambda caution aging, or per-stop caution-vs-green pit-loss logic from race_model")

    emit("")
    emit("=" * 100)
    emit("Step 3 - Structural gap")
    emit("=" * 100)
    emit("race_model/monte_carlo is doing both:")
    emit("1) More detailed physical computation per strategy (driver-level lap simulation with degradation/ageing/pit-loss rules).")
    emit("2) Python-loop execution over iterations, drivers, and laps rather than vectorizing a strategy batch.")
    emit("Therefore the 86-second step 2h result was not a tractability proof for the full package physics; it benchmarked a simpler vectorized surrogate.")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Recommendation (no implementation)")
    emit("=" * 100)
    emit("Direction: rewrite Monte Carlo evaluation to vectorize across a strategy-batch dimension sharing the same schedule draws,")
    emit("but do not assume step 2h code is physically equivalent as-is. Use step 2h as an array-shape/template reference only,")
    emit("then port the real race_model physics into vectorized kernels so optimiser.py can search at locked sizes.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("")
    emit(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
