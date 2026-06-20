# scripts/run_phase3.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import json
import pickle
import pandas as pd
import lightgbm as lgb

from src.monitoring.drift_simulator import generate_all_scenarios
from src.monitoring.drift_monitor   import SlidingWindowNDCGMonitor
from src.monitoring.evidently_drift import run_evidently_drift
from src.monitoring.drift_logger    import log_drift_events_to_mlflow
from src.monitoring.drift_chart     import plot_drift_timeline


def run_phase3(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg['data']['processed_path']
    feature_cols   = cfg['ranker']['feature_cols']

    model_path = os.path.join(processed_path, 'baseline_model.txt')
    model      = lgb.Booster(model_file=model_path)

    for speed in ["sudden", "gradual", "slow"]:
        p = os.path.join(processed_path, f'sim_{speed}.parquet')
        if not os.path.exists(p):
            generate_all_scenarios(config_path)
            break

    results = {}

    for scenario in ["sudden", "gradual", "slow"]:
        print(f"\n{'='*50}")
        print(f"Running drift detection: scenario='{scenario}'")
        print('='*50)

        df  = pd.read_parquet(os.path.join(processed_path, f'sim_{scenario}.parquet'))
        n   = len(df)

        monitor = SlidingWindowNDCGMonitor(
            model          = model,
            feature_cols   = feature_cols,
            window_size    = 200,
            step_size      = 50,
            threshold      = 0.08,
            warmup_windows = 3
        )
        drift_events = monitor.run(df)

        log_path = f"logs/drift_log_{scenario}.json"
        os.makedirs("logs", exist_ok=True)
        monitor.save_log(log_path)

        ref_df     = df.iloc[:n//2]
        cur_df     = df.iloc[n//2:]
        ev_summary = run_evidently_drift(
            ref_df, cur_df, feature_cols,
            output_path = f"logs/evidently_{scenario}.html",
            json_path   = f"logs/evidently_{scenario}_summary.json"
        )

        log_drift_events_to_mlflow(drift_events, ev_summary, scenario, config_path)

        plot_drift_timeline(
            log_path = log_path,
            out_path = f"logs/drift_timeline_{scenario}.html"
        )

        results[scenario] = {
            "n_drift_events":  len(drift_events),
            "dataset_drift":   ev_summary['dataset_drift'],
            "drifted_features": ev_summary['number_of_drifted']
        }

    print("\n\nPhase 3 Summary:")
    print("-" * 40)
    for sc, r in results.items():
        print(f"{sc:10s}: {r['n_drift_events']} drift events | "
              f"Evidently drift={r['dataset_drift']} | "
              f"drifted_features={r['drifted_features']}")

    return results


if __name__ == "__main__":
    run_phase3()