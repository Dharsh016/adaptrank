import os
import json
import pytest
import pandas as pd
import lightgbm as lgb
import yaml


def test_simulation_files_exist():
    for speed in ["sudden", "gradual", "slow"]:
        path = f"data/processed/sim_{speed}.parquet"
        assert os.path.exists(path), f"Missing {path} — run scripts/run_phase3.py"


def test_simulation_has_drift_flag():
    df = pd.read_parquet("data/processed/sim_sudden.parquet")
    assert 'drift_injected' in df.columns
    assert df['drift_injected'].sum() > 0


def test_monitor_detects_drift_on_sudden():
    import yaml, lightgbm as lgb
    from src.monitoring.drift_monitor import SlidingWindowNDCGMonitor

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = lgb.Booster(
        model_file=os.path.join(cfg['data']['processed_path'], 'baseline_model.txt')
    )
    df = pd.read_parquet("data/processed/sim_sudden.parquet")

    monitor = SlidingWindowNDCGMonitor(
        model=model,
        feature_cols=cfg['ranker']['feature_cols'],
        window_size=200, step_size=50, threshold=0.08, warmup_windows=3
    )
    events = monitor.run(df)
    assert len(events) >= 1, "Sudden drift scenario must trigger at least 1 drift event"


def test_drift_log_saved():
    from src.monitoring.drift_monitor import SlidingWindowNDCGMonitor
    import yaml, lightgbm as lgb

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = lgb.Booster(
        model_file=os.path.join(cfg['data']['processed_path'], 'baseline_model.txt')
    )
    df = pd.read_parquet("data/processed/sim_sudden.parquet")
    monitor = SlidingWindowNDCGMonitor(
        model=model, feature_cols=cfg['ranker']['feature_cols'],
        window_size=200, step_size=50, threshold=0.08, warmup_windows=3
    )
    monitor.run(df)
    os.makedirs("logs", exist_ok=True)
    monitor.save_log("logs/drift_log_sudden.json")
    assert os.path.exists("logs/drift_log_sudden.json")

    with open("logs/drift_log_sudden.json") as f:
        log = json.load(f)
    assert "window_log"   in log
    assert "drift_events" in log
    assert len(log["window_log"]) > 0


def test_evidently_runs():
    import yaml
    from src.monitoring.evidently_drift import run_evidently_drift

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    df      = pd.read_parquet("data/processed/sim_sudden.parquet")
    n       = len(df)
    summary = run_evidently_drift(
        df.iloc[:n//2], df.iloc[n//2:],
        cfg['ranker']['feature_cols'],
        output_path="logs/test_evidently.html",
        json_path="logs/test_evidently_summary.json"
    )
    assert "dataset_drift"     in summary
    assert "number_of_drifted" in summary