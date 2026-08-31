"""gamma x kappa sweep, on the unified harness. Replaces v1/v2 status_scaling.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # sandbox layout; use parents[3] under src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.base import run_cli
from harness.experiments import StatusScaling

if __name__ == "__main__":
    run_cli(StatusScaling())
