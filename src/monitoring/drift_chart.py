# src/monitoring/drift_chart.py

import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_drift_timeline(
    log_path:   str = "logs/drift_log.json",
    out_path:   str = "logs/drift_timeline.html"
):
    """
    Render an interactive Plotly chart showing:
    - NDCG@10 across all windows
    - Baseline reference line
    - Red markers where drift was detected
    """
    with open(log_path) as f:
        log = json.load(f)

    window_log    = log['window_log']
    drift_events  = log['drift_events']
    baseline_ndcg = log.get('baseline_ndcg', None)
    threshold     = log.get('threshold', 0.08)

    windows = [w['window_idx'] for w in window_log]
    ndcg10s = [w['ndcg10']     for w in window_log]

    drift_windows = []
    drift_ndcgs   = []
    for event in drift_events:
        # Find matching window index from log
        for w in window_log:
            if w['start'] == event['window_start']:
                drift_windows.append(w['window_idx'])
                drift_ndcgs.append(event['ndcg10_current'])
                break

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(
        x=windows, y=ndcg10s,
        mode='lines+markers',
        name='NDCG@10 per window',
        line=dict(color='steelblue', width=2),
        marker=dict(size=4)
    ))

    if baseline_ndcg:
        fig.add_hline(
            y=baseline_ndcg, line_dash='dash',
            line_color='green',
            annotation_text=f"Baseline {baseline_ndcg:.3f}"
        )
        fig.add_hline(
            y=baseline_ndcg - threshold, line_dash='dot',
            line_color='orange',
            annotation_text=f"Drift threshold -{threshold}"
        )

    if drift_windows:
        fig.add_trace(go.Scatter(
            x=drift_windows, y=drift_ndcgs,
            mode='markers',
            name='Drift detected',
            marker=dict(color='red', size=12, symbol='x')
        ))

    fig.update_layout(
        title='AdaptRank — Ranking Quality Drift Timeline',
        xaxis_title='Window index',
        yaxis_title='NDCG@10',
        legend=dict(x=0.01, y=0.01),
        height=450
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    print(f"Drift timeline chart saved → {out_path}")
    return fig


if __name__ == "__main__":
    plot_drift_timeline()