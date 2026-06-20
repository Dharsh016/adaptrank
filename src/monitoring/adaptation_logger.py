# src/monitoring/adaptation_logger.py
# Logging for model adaptation and recovery events
# src/monitoring/adaptation_logger.py

import os
import json
import mlflow
import yaml
import pandas as pd
from dataclasses import asdict


def log_adaptation_to_mlflow(
    all_results:  dict,
    table:        pd.DataFrame,
    config_path:  str = "config.yaml"
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    # One MLflow run per scenario × refit_window combination
    for scenario, rw_dict in all_results.items():
        for refit_window, results in rw_dict.items():
            run_name = f"adapt-{scenario}-rw{refit_window}"

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("scenario",         scenario)
                mlflow.log_param("refit_window_size", refit_window)
                mlflow.log_param("n_drift_events",    len(results))

                if results:
                    latencies   = [r.interactions_to_recover for r in results]
                    recovered   = [r.recovered for r in results]
                    ndcg_deltas = [r.ndcg10_delta for r in results]

                    import numpy as np
                    mlflow.log_metric("avg_latency",    round(float(np.mean(latencies)), 1))
                    mlflow.log_metric("min_latency",    int(np.min(latencies)))
                    mlflow.log_metric("max_latency",    int(np.max(latencies)))
                    mlflow.log_metric("recovery_rate",  round(sum(recovered)/len(recovered), 3))
                    mlflow.log_metric("avg_ndcg_delta", round(float(np.mean(ndcg_deltas)), 4))

                    for i, r in enumerate(results):
                        mlflow.log_metric("latency_event",    r.interactions_to_recover, step=i)
                        mlflow.log_metric("ndcg_after_refit", r.ndcg10_after,            step=i)

                # Log table artifact
                if os.path.exists("logs/latency_benchmark_table.csv"):
                    mlflow.log_artifact("logs/latency_benchmark_table.csv")

    print("MLflow: all adaptation runs logged.")