# src/agents/profile_agent.py

import os
import pickle
import numpy as np
import pandas as pd
from src.agents.state import AdaptRankState


def profile_agent_node(state: AdaptRankState) -> AdaptRankState:
    """
    Node 2: Computes learner profile shift around the first drift event.
    Compares pre-drift and post-drift profile centroids to describe
    how the learner population moved in embedding space.
    Writes: learner_profiles (summary), profile_shift_summary
    """
    print("[Profile Agent] Computing learner profile shifts...")

    if not state.get('drift_detected'):
        print("[Profile Agent] No drift detected — skipping profile analysis.")
        return {
            **state,
            'profile_shift_summary': "No drift detected. Profile analysis skipped.",
            'learner_profiles': {}
        }

    import yaml
    with open(state['config_path']) as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg['data']['processed_path']
    emb_path       = os.path.join(processed_path, 'course_embeddings.pkl')

    with open(emb_path, 'rb') as f:
        course_emb_map = pickle.load(f)

    df = pd.read_parquet(state['df_path'])
    feature_cols = state['feature_cols']

    # Use first drift event to split pre/post
    first_event   = state['drift_events'][0]
    drift_at      = first_event['window_start']

    pre_df  = df.iloc[:drift_at].copy()
    post_df = df.iloc[drift_at:drift_at + 500].copy()

    def mean_profile(subset_df):
        vecs = []
        for _, row in subset_df.iterrows():
            emb = course_emb_map.get(row['code_module'])
            if emb is not None:
                vecs.append(emb)
        return np.mean(vecs, axis=0) if vecs else None

    pre_centroid  = mean_profile(pre_df)
    post_centroid = mean_profile(post_df)

    # Compute feature-level shift for summary
    feature_shifts = {}
    for col in feature_cols:
        if col in df.columns:
            pre_mean  = pre_df[col].fillna(0).mean()
            post_mean = post_df[col].fillna(0).mean()
            pct_change = ((post_mean - pre_mean) / (abs(pre_mean) + 1e-9)) * 100
            feature_shifts[col] = round(pct_change, 1)

    # Sort by absolute shift magnitude
    top_shifts = sorted(feature_shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    # Cosine similarity between pre and post centroids
    cosine_sim = None
    if pre_centroid is not None and post_centroid is not None:
        denom = (np.linalg.norm(pre_centroid) * np.linalg.norm(post_centroid))
        if denom > 0:
            cosine_sim = float(np.dot(pre_centroid, post_centroid) / denom)

    shift_lines = [f"{col}: {pct:+.1f}%" for col, pct in top_shifts]
    sim_str = f"{cosine_sim:.4f}" if cosine_sim is not None else "N/A"
    summary = (
        f"Drift detected at row {drift_at}. "
        f"Top feature shifts: {', '.join(shift_lines)}. "
        f"Profile centroid cosine similarity pre/post drift: {sim_str}. "
        f"Total drift events: {len(state['drift_events'])}."
    )

    print(f"[Profile Agent] {summary}")

    return {
        **state,
        'profile_shift_summary': summary,
        'learner_profiles': {
            'cosine_similarity': cosine_sim,
            'top_feature_shifts': dict(top_shifts),
            'n_pre_rows': len(pre_df),
            'n_post_rows': len(post_df)
        }
    }