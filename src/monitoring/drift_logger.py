# src/monitoring/drift_logger.py

import os
import json
import mlflow
import yaml
from dataclasses import asdict
from src.monitoring.drift_monitor import DriftEvent


def log_drift_events_to_mlflow(
    drift_events:   list,
    evidently_summary: dict,
    scenario_name:  str,
    config_path:    str = "config.yaml"
):
    """Log all drift events and Evidently summary to a single MLflow run."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"drift-detection-{scenario_name}"):
        mlflow.log_param("scenario",        scenario_name)
        mlflow.log_param("n_drift_events",  len(drift_events))
        mlflow.log_param("evidently_drift", evidently_summary.get('dataset_drift'))

        mlflow.log_metric("drift_share",
                          evidently_summary.get('drift_share', 0.0))
        mlflow.log_metric("n_drifted_features",
                          evidently_summary.get('number_of_drifted', 0))

        for i, event in enumerate(drift_events):
            d = asdict(event) if not isinstance(event, dict) else event
            mlflow.log_metric(f"event_{i}_ndcg_drop",    d['ndcg10_drop'],    step=i)
            mlflow.log_metric(f"event_{i}_ndcg_current", d['ndcg10_current'], step=i)

        # Log the full event log as artifact
        events_path = f"logs/drift_events_{scenario_name}.json"
        os.makedirs("logs", exist_ok=True)
        with open(events_path, 'w') as f:
            json.dump([asdict(e) if not isinstance(e, dict) else e
                       for e in drift_events], f, indent=2)
        mlflow.log_artifact(events_path)

        evidently_html = "logs/evidently_drift_report.html"
        if os.path.exists(evidently_html):
            mlflow.log_artifact(evidently_html)

    print(f"MLflow: logged {len(drift_events)} drift events for scenario '{scenario_name}'")