# src/monitoring/recovery_chart.py
# Visualization for model recovery after drift
# src/monitoring/recovery_chart.py

import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_recovery_curves(
    results_path: str = "logs/adaptation_results.json",
    out_path:     str = "logs/recovery_curves.html"
):
    with open(results_path) as f:
        data = json.load(f)

    scenarios = list(data.keys())
    refit_windows = ["200", "500"]

    fig = make_subplots(
        rows=1, cols=len(scenarios),
        subplot_titles=[f"Scenario: {s}" for s in scenarios],
        shared_yaxes=True
    )

    colors = {"200": "steelblue", "500": "darkorange"}

    for col_idx, scenario in enumerate(scenarios, start=1):
        for rw in refit_windows:
            results = data.get(scenario, {}).get(rw, [])
            if not results:
                continue

            latencies   = [r['interactions_to_recover'] for r in results]
            ndcg_deltas = [r['ndcg10_delta'] for r in results]
            recovered   = [r['recovered'] for r in results]

            # Plot each event as a point: x=latency, y=ndcg_delta
            fig.add_trace(
                go.Scatter(
                    x    = latencies,
                    y    = ndcg_deltas,
                    mode = 'markers',
                    name = f"refit_window={rw}",
                    marker = dict(
                        color  = [colors[rw]] * len(latencies),
                        size   = 10,
                        symbol = ['circle' if r else 'x' for r in recovered]
                    ),
                    showlegend = (col_idx == 1),
                    legendgroup = rw
                ),
                row=1, col=col_idx
            )

    fig.update_xaxes(title_text="Interactions to recover (latency)")
    fig.update_yaxes(title_text="NDCG@10 delta after refit", col=1)
    fig.update_layout(
        title = "Ranking Adaptation Latency — Recovery Curves",
        height = 450,
        legend = dict(x=0.01, y=0.99)
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    print(f"Recovery curves chart → {out_path}")
    return fig


if __name__ == "__main__":
    plot_recovery_curves()