"""Entry point. The whole experiment is now the class; this is the shim."""
from harness.base import run_cli
from harness.experiments import StatusScaling

if __name__ == "__main__":
    run_cli(StatusScaling())
