# src/agents/agent_logger.py

import os
import json
import mlflow
import yaml


def log_agent_run(final_state: dict, config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])

    scenario = final_state.get('scenario', 'unknown')

    with mlflow.start_run(run_name=f"agent-pipeline-{scenario}"):
        mlflow.log_param("scenario",       scenario)
        mlflow.log_param("n_drift_events", len(final_state.get('drift_events') or []))
        mlflow.log_param("refit_window",   final_state.get('refit_window_used', 200))

        mlflow.log_metric("baseline_ndcg",  final_state.get('baseline_ndcg', 0) or 0)
        mlflow.log_metric("avg_latency",    final_state.get('avg_latency', 0) or 0)
        mlflow.log_metric("recovery_rate",  final_state.get('recovery_rate', 0) or 0)

        # Save report as artifact
        os.makedirs("logs", exist_ok=True)
        report_path = f"logs/agent_report_{scenario}.txt"
        with open(report_path, 'w') as f:
            f.write(final_state.get('explanation_report', ''))
        mlflow.log_artifact(report_path)

        # Save full state as JSON artifact
        state_path = f"logs/agent_state_{scenario}.json"
        serializable = {
            k: v for k, v in final_state.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        with open(state_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        mlflow.log_artifact(state_path)

    print(f"MLflow: agent run logged for scenario '{scenario}'")