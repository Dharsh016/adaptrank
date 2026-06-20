# src/agents/state.py

from typing import TypedDict, Optional, List
from src.monitoring.drift_monitor import DriftEvent
from src.ranker.adapter import AdaptationResult


class AdaptRankState(TypedDict):
    """
    Shared state passed between all 4 agents in the LangGraph pipeline.
    Each agent reads from state and writes its output back to state.
    """
    # Input — provided at pipeline start
    scenario:          str
    df_path:           str              # path to simulation parquet
    feature_cols:      list
    model_path:        str
    config_path:       str

    # Drift Sentinel output
    drift_events:      Optional[List[dict]]    # list of DriftEvent as dicts
    baseline_ndcg:     Optional[float]
    drift_detected:    Optional[bool]

    # Profile Agent output
    learner_profiles:  Optional[dict]          # sid -> embedding array (as list for JSON)
    profile_shift_summary: Optional[str]       # plain description of profile changes

    # Re-ranker output
    adaptation_results: Optional[List[dict]]   # list of AdaptationResult as dicts
    refit_window_used:  Optional[int]
    avg_latency:       Optional[float]
    recovery_rate:     Optional[float]

    # Explainer output
    explanation_report: Optional[str]          # final LLM-generated narrative
    rubric_scores:      Optional[dict]         # criteria -> score