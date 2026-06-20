# scripts/run_phase5.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.pipeline    import run_pipeline
from src.agents.agent_logger import log_agent_run


def main():
    for scenario in ["sudden", "gradual", "slow"]:
        print(f"\n{'#'*60}")
        print(f"PIPELINE: scenario={scenario}")
        print('#'*60)
        final_state = run_pipeline(scenario)
        log_agent_run(final_state)
        print(f"Done: {scenario}")


if __name__ == "__main__":
    main()