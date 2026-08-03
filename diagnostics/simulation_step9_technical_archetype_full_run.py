from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.run import (  # noqa: E402
    DEFAULT_TECHNICAL_CACHE_PATH,
    run_phase3_technical_first_pass,
)


DIAG_DIR = ROOT / "data" / "diagnostics"
OUTPUT_PATH = DIAG_DIR / "simulation_step9_technical_archetype_full_run_output.txt"


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    result = run_phase3_technical_first_pass(
        output_path=OUTPUT_PATH,
        cache_path=DEFAULT_TECHNICAL_CACHE_PATH,
    )
    print(f"output_file={result['output_path']}")
    print(f"output_file_bytes={result['output_path'].stat().st_size if result['output_path'].exists() else 0}")
    print(f"cache_file={result['cache_path']}")
    print(f"cache_file_bytes={result['cache_path'].stat().st_size if result['cache_path'].exists() else 0}")


if __name__ == "__main__":
    main()
