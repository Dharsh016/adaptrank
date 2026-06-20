# src/agents/drift_sentinel.py

import pickle
import yaml
import pandas as pd
import lightgbm as lgb
from dataclasses import asdict
from src.agents.state import AdaptRankState
from src.monitoring.drift_monitor import SlidingWindowNDCGMonitor


def drift_sentinel_node(state: AdaptRankState) -> AdaptRankState:
    """
    Node 1: Runs the sliding-window NDCG monitor on the scenario data.
    Detects drift events and establishes baseline NDCG.
    Writes: drift_events, baseline_ndcg, drift_detected
    """
    print("[Drift Sentinel] Starting drift detection...")

    with open(state['config_path']) as f:
        import yaml
        cfg = yaml.safe_load(f)

    df    = pd.read_parquet(state['df_path'])
    model = lgb.Booster(model_file=state['model_path'])

    monitor = SlidingWindowNDCGMonitor(
        model          = model,
        feature_cols   = state['feature_cols'],
        window_size    = 200,
        step_size      = 50,
        threshold      = 0.08,
        warmup_windows = 3
    )
    drift_events   = monitor.run(df)
    baseline_ndcg  = monitor.baseline_ndcg

    print(f"[Drift Sentinel] Detected {len(drift_events)} drift events. "
          f"Baseline NDCG@10={baseline_ndcg:.4f}")

    return {
        **state,
        'drift_events':   [asdict(e) for e in drift_events],
        'baseline_ndcg':  baseline_ndcg,
        'drift_detected': len(drift_events) > 0
    }