# src/agents/reranker_agent.py

import pandas as pd
import lightgbm as lgb
from src.agents.state import AdaptRankState
from src.monitoring.drift_monitor import SlidingWindowNDCGMonitor
from src.ranker.adapter import AdaptationEngine
from dataclasses import asdict


def reranker_agent_node(state: AdaptRankState) -> AdaptRankState:
    """
    Node 3: Triggers LightGBM refit using the AdaptationEngine.
    Uses refit_window=200 for the agent pipeline (speed).
    Writes: adaptation_results, avg_latency, recovery_rate, refit_window_used
    """
    print("[Re-ranker Agent] Starting adaptation engine...")

    if not state.get('drift_detected'):
        print("[Re-ranker Agent] No drift — skipping refit.")
        return {
            **state,
            'adaptation_results': [],
            'avg_latency': 0.0,
            'recovery_rate': 1.0,
            'refit_window_used': 200
        }

    import yaml
    with open(state['config_path']) as f:
        cfg = yaml.safe_load(f)

    df    = pd.read_parquet(state['df_path'])
    model = lgb.Booster(model_file=state['model_path'])

    from src.monitoring.drift_monitor import DriftEvent
    from dataclasses import fields

    # Reconstruct DriftEvent objects from state dicts
    drift_events = []
    for d in state['drift_events']:
        drift_events.append(DriftEvent(**d))

    engine = AdaptationEngine(
        model              = model,
        feature_cols       = state['feature_cols'],
        cfg                = cfg,
        refit_window_size  = 200,
        recovery_threshold = 0.05,
        window_size        = 200,
        step_size          = 50
    )

    results = engine.run(
        df, drift_events, state['baseline_ndcg'], state['scenario']
    )

    if results:
        import numpy as np
        latencies     = [r.interactions_to_recover for r in results]
        recovered     = [r.recovered for r in results]
        avg_latency   = float(np.mean(latencies))
        recovery_rate = float(sum(recovered) / len(recovered))
    else:
        avg_latency   = 0.0
        recovery_rate = 1.0

    print(f"[Re-ranker Agent] Adaptation complete. "
          f"avg_latency={avg_latency:.1f}, recovery_rate={recovery_rate:.3f}")

    return {
        **state,
        'adaptation_results': [asdict(r) for r in results],
        'avg_latency':        avg_latency,
        'recovery_rate':      recovery_rate,
        'refit_window_used':  200
    }