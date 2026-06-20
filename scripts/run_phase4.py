# scripts/run_phase4.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ranker.latency_benchmark   import run_latency_benchmark, build_benchmark_table, save_benchmark
from src.monitoring.adaptation_logger import log_adaptation_to_mlflow
from src.monitoring.recovery_chart    import plot_recovery_curves


def main():
    print("Running Ranking Adaptation Latency benchmark...")
    all_results = run_latency_benchmark()

    print("\nBuilding benchmark table...")
    table = build_benchmark_table(all_results)
    save_benchmark(all_results, table)

    print("\nLogging to MLflow...")
    log_adaptation_to_mlflow(all_results, table)

    print("\nGenerating recovery curves chart...")
    plot_recovery_curves()

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()