# src/ranker/latency_benchmark.py
# Latency benchmarking for inference performance
# src/ranker/latency_benchmark.py

import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import asdict

from src.monitoring.drift_monitor import SlidingWindowNDCGMonitor, DriftEvent
from src.ranker.adapter import AdaptationEngine


def run_latency_benchmark(config_path="config.yaml") -> dict:
    """
    Run adaptation across all 3 scenarios × 2 refit window sizes.
    Produces the Ranking Adaptation Latency table.

    Returns nested dict:
      results[scenario][refit_window] = list of AdaptationResult
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg['data']['processed_path']
    feature_cols   = cfg['ranker']['feature_cols']

    model_path = os.path.join(processed_path, 'baseline_model.txt')
    model      = lgb.Booster(model_file=model_path)

    scenarios     = ["sudden", "gradual", "slow"]
    refit_windows = [200, 500]

    all_results = {}

    for scenario in scenarios:
        all_results[scenario] = {}
        df = pd.read_parquet(
            os.path.join(processed_path, f'sim_{scenario}.parquet')
        )

        # Re-run monitor to get drift events + baseline for this scenario
        monitor = SlidingWindowNDCGMonitor(
            model          = model,
            feature_cols   = feature_cols,
            window_size    = 200,
            step_size      = 50,
            threshold      = 0.08,
            warmup_windows = 3
        )
        drift_events   = monitor.run(df)
        baseline_ndcg  = monitor.baseline_ndcg

        if not drift_events:
            print(f"WARNING: No drift events detected for scenario '{scenario}'. "
                  f"Lower threshold or check simulation.")
            for rw in refit_windows:
                all_results[scenario][rw] = []
            continue

        for refit_window in refit_windows:
            print(f"\n{'='*55}")
            print(f"Scenario={scenario}  refit_window={refit_window}")
            print('='*55)

            engine = AdaptationEngine(
                model              = model,
                feature_cols       = feature_cols,
                cfg                = cfg,
                refit_window_size  = refit_window,
                recovery_threshold = 0.05,
                window_size        = 200,
                step_size          = 50,
            )
            results = engine.run(df, drift_events, baseline_ndcg, scenario)
            all_results[scenario][refit_window] = results

    return all_results


def build_benchmark_table(all_results: dict) -> pd.DataFrame:
    """
    Convert all_results into the Ranking Adaptation Latency table.

    Columns: scenario, refit_window, avg_latency, min_latency,
             max_latency, recovery_rate, avg_ndcg_delta
    """
    rows = []
    for scenario, rw_dict in all_results.items():
        for refit_window, results in rw_dict.items():
            if not results:
                rows.append({
                    'scenario':       scenario,
                    'refit_window':   refit_window,
                    'avg_latency':    None,
                    'min_latency':    None,
                    'max_latency':    None,
                    'recovery_rate':  0.0,
                    'avg_ndcg_delta': None,
                    'n_events':       0
                })
                continue

            latencies   = [r.interactions_to_recover for r in results]
            recovered   = [r.recovered for r in results]
            ndcg_deltas = [r.ndcg10_delta for r in results]

            rows.append({
                'scenario':       scenario,
                'refit_window':   refit_window,
                'avg_latency':    round(np.mean(latencies), 1),
                'min_latency':    int(np.min(latencies)),
                'max_latency':    int(np.max(latencies)),
                'recovery_rate':  round(sum(recovered) / len(recovered), 3),
                'avg_ndcg_delta': round(np.mean(ndcg_deltas), 4),
                'n_events':       len(results)
            })

    df_table = pd.DataFrame(rows)
    return df_table


def save_benchmark(all_results: dict, table: pd.DataFrame, config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("logs", exist_ok=True)

    # Save full results as JSON
    serializable = {}
    for scenario, rw_dict in all_results.items():
        serializable[scenario] = {}
        for rw, results in rw_dict.items():
            serializable[scenario][str(rw)] = [asdict(r) for r in results]

    with open("logs/adaptation_results.json", 'w') as f:
        json.dump(serializable, f, indent=2)

    # Save table as CSV
    table.to_csv("logs/latency_benchmark_table.csv", index=False)

    print("\n" + "="*55)
    print("RANKING ADAPTATION LATENCY BENCHMARK TABLE")
    print("="*55)
    print(table.to_string(index=False))
    print("\nSaved → logs/adaptation_results.json")
    print("Saved → logs/latency_benchmark_table.csv")


if __name__ == "__main__":
    all_results = run_latency_benchmark()
    table       = build_benchmark_table(all_results)
    save_benchmark(all_results, table)