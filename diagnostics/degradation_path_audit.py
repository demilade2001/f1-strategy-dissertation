from __future__ import annotations

import json
import math
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__"}
FIGURE_NAME = "tyre_degradation_curves_hard"
BUILD_FN = "build_tyre_degradation"


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path.relative_to(root)):
            continue
        yield path


def grep_like(root: Path, pattern: str):
    matches = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                matches.append((path, line_no, line))
    return matches


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def extract_notebook_code_cell_block(ipynb_path: Path) -> list[str]:
    raw = json.loads(ipynb_path.read_text(encoding="utf-8"))
    for cell in raw.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "tyre_degradation_curves_" in src and "np.polyfit" in src and "fuel_corrected_laptime" in src:
            return src.splitlines()
    return []


def print_context(path: Path, line_no: int, radius: int = 10) -> None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    print(f"\nContext: {path.relative_to(REPO_ROOT)}:{line_no} (lines {start}-{end})")
    for idx in range(start, end + 1):
        print(f"{idx:5d}: {lines[idx - 1]}")


def get_build_like_eligibility(df: pd.DataFrame) -> pd.Series:
    deg_eligible = pd.Series(True, index=df.index)
    deg_eligible &= (df["compound_known"] == True)
    deg_eligible &= (df["IsAccurate"] == True)
    deg_eligible &= (df["sc_active"] == False)
    deg_eligible &= (df["vsc_active"] == False)
    deg_eligible &= (df["PitInTime"].isna())
    deg_eligible &= (df["PitOutTime"].isna())
    wet_compounds = {"INTERMEDIATE", "WET"}
    deg_eligible &= ~df["TyreCompound"].isin(wet_compounds)

    for (driver, round_num), group in df[deg_eligible].groupby(["Driver", "Round"]):
        median_time = group["LapTime_s"].median()
        threshold = median_time * 1.07
        outlier_indices = group[group["LapTime_s"] > threshold].index
        deg_eligible.loc[outlier_indices] = False

    return deg_eligible


def fit_like_build(group: pd.DataFrame) -> tuple[float, str]:
    if len(group) < 5:
        return (float("nan"), "insufficient")

    x = group["TyreLife"].values.astype(float)
    y = group["LapTime_s"].values.astype(float)

    if np.isnan(x).any() or np.isnan(y).any():
        return (float("nan"), "invalid_data")

    try:
        coeffs_linear = np.polyfit(x, y, 1)
        y_pred_linear = np.polyval(coeffs_linear, x)
        ss_res_linear = np.sum((y - y_pred_linear) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
    except (np.linalg.LinAlgError, ValueError):
        r2_linear = np.nan
        coeffs_linear = None

    try:
        coeffs_quadratic = np.polyfit(x, y, 2)
        y_pred_quadratic = np.polyval(coeffs_quadratic, x)
        ss_res_quadratic = np.sum((y - y_pred_quadratic) ** 2)
        r2_quadratic = 1 - (ss_res_quadratic / ss_tot) if ss_tot > 0 else 0
    except (np.linalg.LinAlgError, ValueError):
        r2_quadratic = np.nan
        coeffs_quadratic = None

    if (
        not np.isnan(r2_quadratic)
        and not np.isnan(r2_linear)
        and r2_quadratic > r2_linear + 0.02
        and coeffs_quadratic is not None
    ):
        a = coeffs_quadratic[0]
        b = coeffs_quadratic[1]
        mean_tyre_life = x.mean()
        return (float(2 * a * mean_tyre_life + b), "quadratic")

    if coeffs_linear is not None:
        return (float(coeffs_linear[0]), "linear")

    return (float("nan"), "invalid_fit")


def figure_path_metrics(df: pd.DataFrame, event_name: str, compound: str) -> tuple[int, float]:
    curve_base = df[
        df["TyreCompound"].isin(["SOFT", "MEDIUM", "HARD"])
        & df["TyreLife"].notna()
        & df["fuel_corrected_laptime"].notna()
    ].copy()
    sub = curve_base[(curve_base["TyreCompound"] == compound) & (curve_base["EventName"] == event_name)].copy()
    x = pd.to_numeric(sub["TyreLife"], errors="coerce")
    y = pd.to_numeric(sub["fuel_corrected_laptime"], errors="coerce")
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    slope = float(np.polyfit(x, y, 1)[0]) if n >= 5 else float("nan")
    return n, slope


def build_path_metrics(df: pd.DataFrame, event_name: str, compound: str) -> tuple[int, float, str]:
    deg_eligible = get_build_like_eligibility(df)
    group = df[deg_eligible & (df["EventName"] == event_name) & (df["TyreCompound"] == compound)].copy()
    n = len(group)
    slope, model = fit_like_build(group)
    return n, slope, model


def fmt_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def main() -> None:
    print_section("Step 1 - grep -rn \"tyre_degradation_curves_hard\" . (excluding .git/.venv/venv)")
    fig_matches = grep_like(REPO_ROOT, FIGURE_NAME)
    if not fig_matches:
        print("No matches found.")
    else:
        for path, line_no, line in fig_matches:
            rel = path.relative_to(REPO_ROOT)
            print(f"{rel}:{line_no}:{line}")

    print_section("Step 2 - Verbatim filter + regression block for hard-curve figure path")
    notebook_path = REPO_ROOT / "notebooks" / "01_eda.ipynb"
    block = extract_notebook_code_cell_block(notebook_path)
    if not block:
        print("No matching notebook code cell found.")
    else:
        print(f"Source file: {notebook_path.relative_to(REPO_ROOT)}")
        for line in block:
            print(line)

    print_section("Step 3 - grep -rn \"build_tyre_degradation\" . + call-site/import contexts")
    fn_matches = grep_like(REPO_ROOT, BUILD_FN)
    for path, line_no, line in fn_matches:
        rel = path.relative_to(REPO_ROOT)
        print(f"{rel}:{line_no}:{line}")

    context_targets = []
    for path in iter_text_files(REPO_ROOT):
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(alias.name == BUILD_FN for alias in node.names):
                    context_targets.append((path, node.lineno))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == BUILD_FN:
                    context_targets.append((path, node.lineno))
                elif isinstance(node.func, ast.Attribute) and node.func.attr == BUILD_FN:
                    context_targets.append((path, node.lineno))

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for item in context_targets:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    context_targets = deduped

    if not context_targets:
        print("\nNo import/call occurrences found beyond definition/comments.")
    else:
        for path, line_no in context_targets:
            print_context(path, line_no, radius=10)

    print_section("Step 4 - source_code/simulation.py check")
    src_dir = REPO_ROOT / "src"
    src_entries = sorted([p.name for p in src_dir.iterdir()])
    print("ls source_code/")
    for name in src_entries:
        print(name)

    simulation_path = src_dir / "simulation.py"
    if simulation_path.exists():
        print("\nFile exists: source_code/simulation.py")
        sim_lines = simulation_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        degrad_hits = [i for i, ln in enumerate(sim_lines, start=1) if "degrad" in ln.lower()]
        if not degrad_hits:
            print("No matches for grep -n -i \"degrad\" source_code/simulation.py")
        else:
            for ln in degrad_hits:
                print_context(simulation_path, ln, radius=5)
    else:
        print("simulation.py not found; see ls source_code/ output above.")

    print_section("Step 5 - Two-path N/slope comparison table")
    base_df_path = REPO_ROOT / "data" / "processed" / "base_df.csv"
    df = pd.read_csv(base_df_path)

    for col in ["sc_active", "vsc_active"]:
        df[col] = df[col].infer_objects(copy=False).fillna(False).astype(bool)

    combos = [
        ("British Grand Prix", "SOFT"),
        ("Hungarian Grand Prix", "HARD"),
    ]

    rows = []
    for event_name, compound in combos:
        fig_n, fig_slope = figure_path_metrics(df, event_name, compound)
        build_n, build_slope, build_model = build_path_metrics(df, event_name, compound)
        rows.append(
            {
                "EventName": event_name,
                "Compound": compound,
                "FigurePath_N": fig_n,
                "FigurePath_Slope": fmt_float(fig_slope),
                "BuildPath_N": build_n,
                "BuildPath_Slope": fmt_float(build_slope),
                "BuildPath_Model": build_model,
            }
        )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
