# src/monitoring/evidently_drift.py

import os
import json
import pandas as pd
import numpy as np
import yaml

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


def run_evidently_drift(
    reference_df:  pd.DataFrame,
    current_df:    pd.DataFrame,
    feature_cols:  list,
    output_path:   str = "logs/evidently_drift_report.html",
    json_path:     str = "logs/evidently_drift_summary.json"
) -> dict:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ref = reference_df[feature_cols].fillna(0).reset_index(drop=True)
    cur = current_df[feature_cols].fillna(0).reset_index(drop=True)

    # ColumnMapping no longer needed for pure numeric feature drift
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    report.save_html(output_path)

    report_dict   = report.as_dict()
    drift_results = report_dict['metrics'][0]['result']

    summary = {
        "dataset_drift":     drift_results.get('dataset_drift', False),
        "drift_share":       round(drift_results.get('share_of_drifted_columns', 0.0), 4),
        "number_of_drifted": drift_results.get('number_of_drifted_columns', 0),
        "number_of_columns": drift_results.get('number_of_columns', 0),
        "per_feature":       {}
    }

    col_results = drift_results.get('drift_by_columns', {})
    for col, col_data in col_results.items():
        summary['per_feature'][col] = {
            "drift_detected": col_data.get('drift_detected', False),
            "stattest":       col_data.get('stattest_name', ''),
            "p_value":        round(col_data.get('p_value', 1.0), 4),
        }

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Evidently report saved → {output_path}")
    print(f"Dataset drift: {summary['dataset_drift']}  "
          f"Drifted features: {summary['number_of_drifted']}/{summary['number_of_columns']}")
    return summary


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(
        os.path.join(cfg['data']['processed_path'], 'sim_sudden.parquet')
    )
    feature_cols = cfg['ranker']['feature_cols']
    n = len(df)
    run_evidently_drift(df.iloc[:n//2], df.iloc[n//2:], feature_cols)