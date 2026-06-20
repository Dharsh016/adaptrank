# tests/test_phase4.py

import os
import json
import pytest
import pandas as pd


def test_adaptation_results_exist():
    assert os.path.exists("logs/adaptation_results.json"), \
        "Run scripts/run_phase4.py first"


def test_benchmark_table_exists():
    assert os.path.exists("logs/latency_benchmark_table.csv")


def test_benchmark_table_shape():
    table = pd.read_csv("logs/latency_benchmark_table.csv")
    # 3 scenarios × 2 refit windows = 6 rows
    assert len(table) == 6, f"Expected 6 rows, got {len(table)}"
    for col in ['scenario', 'refit_window', 'avg_latency', 'recovery_rate']:
        assert col in table.columns, f"Missing column: {col}"


def test_sudden_drift_has_results():
    with open("logs/adaptation_results.json") as f:
        data = json.load(f)
    assert "sudden" in data
    # At least one refit window produced results
    has_results = any(
        len(v) > 0 for v in data["sudden"].values()
    )
    assert has_results, "Sudden scenario produced no adaptation results"


def test_recovery_chart_exists():
    assert os.path.exists("logs/recovery_curves.html")


def test_adaptation_engine_runs():
    import yaml
    import lightgbm as lgb
    from src.monitoring.drift_monitor import SlidingWindowNDCGMonitor
    from src.ranker.adapter import AdaptationEngine

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = lgb.Booster(
        model_file=os.path.join(cfg['data']['processed_path'], 'baseline_model.txt')
    )
    df = pd.read_parquet("data/processed/sim_sudden.parquet")
    feature_cols = cfg['ranker']['feature_cols']

    monitor = SlidingWindowNDCGMonitor(
        model=model, feature_cols=feature_cols,
        window_size=200, step_size=50, threshold=0.08, warmup_windows=3
    )
    drift_events  = monitor.run(df)
    baseline_ndcg = monitor.baseline_ndcg

    engine = AdaptationEngine(
        model=model, feature_cols=feature_cols, cfg=cfg,
        refit_window_size=200, recovery_threshold=0.05
    )
    results = engine.run(df, drift_events, baseline_ndcg, "sudden")
    # Engine ran without crashing — results can be empty if no drift detected
    assert isinstance(results, list)