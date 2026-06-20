# src/agents/pipeline.py

import os
import yaml
from langgraph.graph import StateGraph, END
from src.agents.state import AdaptRankState
from src.agents.drift_sentinel  import drift_sentinel_node
from src.agents.profile_agent   import profile_agent_node
from src.agents.reranker_agent  import reranker_agent_node
from src.agents.explainer_agent import explainer_agent_node


def build_pipeline():
    """Build and compile the 4-agent LangGraph pipeline."""
    graph = StateGraph(AdaptRankState)

    graph.add_node("drift_sentinel",  drift_sentinel_node)
    graph.add_node("profile_agent",   profile_agent_node)
    graph.add_node("reranker_agent",  reranker_agent_node)
    graph.add_node("explainer_agent", explainer_agent_node)

    graph.set_entry_point("drift_sentinel")
    graph.add_edge("drift_sentinel", "profile_agent")
    graph.add_edge("profile_agent",  "reranker_agent")
    graph.add_edge("reranker_agent", "explainer_agent")
    graph.add_edge("explainer_agent", END)

    return graph.compile()


def run_pipeline(scenario: str, config_path: str = "config.yaml") -> AdaptRankState:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg['data']['processed_path']

    initial_state: AdaptRankState = {
        'scenario':          scenario,
        'df_path':           os.path.join(processed_path, f'sim_{scenario}.parquet'),
        'feature_cols':      cfg['ranker']['feature_cols'],
        'model_path':        os.path.join(processed_path, 'baseline_model.txt'),
        'config_path':       config_path,
        'drift_events':      None,
        'baseline_ndcg':     None,
        'drift_detected':    None,
        'learner_profiles':  None,
        'profile_shift_summary': None,
        'adaptation_results': None,
        'refit_window_used': None,
        'avg_latency':       None,
        'recovery_rate':     None,
        'explanation_report': None,
        'rubric_scores':     None
    }

    pipeline      = build_pipeline()
    final_state   = pipeline.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    for scenario in ["sudden", "gradual", "slow"]:
        print(f"\n{'#'*60}")
        print(f"Running pipeline for scenario: {scenario}")
        print('#'*60)
        result = run_pipeline(scenario)
        print(f"\nFinal report for '{scenario}':")
        print(result['explanation_report'])