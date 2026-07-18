from pathlib import Path

from src.utils import MIDFIELD_CONSTRUCTORS


# Added seconds to pit-loss window when scanning for nearby rivals.
PROXIMITY_MARGIN_S = 5.0
# Sensitivity sweep values for the rival-detection proximity margin.
PROXIMITY_MARGIN_SENSITIVITY_S = [3.0, 5.0, 8.0]
# Whether Red Bull, Ferrari, and Mercedes are eligible rival candidates.
INCLUDE_BIG_THREE_AS_RIVALS = True
# Caution-lap TyreLife increment multipliers to test in degradation encoding.
LAMBDA_GRID = [1.0, 0.75, 0.5]
# Number of Monte Carlo iterations per simulation run.
N_ITERATIONS = 10_000
# Conservatism rule anchor used for bias-threshold design.
CONSERVATISM_REFERENCE = "REGULATORY_MAX_FEASIBLE_LAP"
# Optional hard cutoff for insufficiency; None keeps continuous r_b reporting.
INSUFFICIENCY_CUT = None
# Global random seed for reproducible stochastic components.
RNG_SEED = 42
# Identifiability epsilon for treating |B - R_b| as effectively zero.
UNIDENTIFIABLE_EPSILON = 1e-6


ROOT = Path(__file__).resolve().parents[2]
# Canonical processed lap-level feature dataset.
BASE_DF_PATH = ROOT / "data" / "processed" / "base_df.csv"
# Per-year degradation fit diagnostics produced by Chapter 3 regeneration.
DEG_RATE_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_full_peryear_stats.csv"
# Calibrated XGBoost test-set predictions used for Phase 3 bias-threshold work.
XGB_TEST_PREDICTIONS_PATH = ROOT / "data" / "models" / "classweight_test_predictions.csv"


# Rivals may be any team including Big Three; only midfield teams may be passed as the
# driver/subject argument to optimiser.py's argmax_strategy once that module exists.
def is_valid_subject(team: str) -> bool:
	return team in MIDFIELD_CONSTRUCTORS