# src/ranker/adapter.py
# Model adapter for handling drift and online learning
# src/ranker/adapter.py

import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass
from sklearn.metrics import ndcg_score


@dataclass
class AdaptationResult:
    """Structured result of one refit cycle."""
    scenario:           str
    drift_window_start: int
    drift_window_end:   int
    ndcg10_before:      float
    ndcg10_after:       float
    ndcg10_delta:       float
    ndcg10_baseline:    float
    interactions_to_recover: int   # Ranking Adaptation Latency value
    refit_window_size:  int
    recovered:          bool       # True if within 5% of baseline


def compute_ndcg10(model, df, feature_cols):
    """Compute mean NDCG@10 across all students in df."""
    if len(df) < 2:
        return 0.0

    X     = df[feature_cols].fillna(0).values
    preds = model.predict(X)
    y     = df['relevance'].values
    sids  = df['id_student'].values

    scores = []
    for sid in np.unique(sids):
        mask = sids == sid
        if mask.sum() < 2:
            continue
        scores.append(
            ndcg_score(y[mask].reshape(1, -1),
                       preds[mask].reshape(1, -1), k=10)
        )
    return float(np.mean(scores)) if scores else 0.0


def refit_model(
    model:        lgb.Booster,
    refit_df:     pd.DataFrame,
    feature_cols: list,
    cfg:          dict
) -> lgb.Booster:
    """
    Incrementally update model on recent interactions only.
    Uses LightGBM refit — adjusts leaf values without rebuilding trees.
    Much faster than full retrain.
    """
    X = refit_df[feature_cols].fillna(0).values
    y = refit_df['relevance'].values

    # Group sizes required by LambdaMART
    groups = refit_df['id_student'].values
    g = pd.Series(groups).value_counts().sort_index().values

    # refit requires a Dataset with group info
    dataset = lgb.Dataset(X, label=y, group=g)
    updated_model = model.refit(X, y, group=g)
    return updated_model


class AdaptationEngine:
    """
    Watches for DriftEvents from the monitor, triggers refit,
    and measures Ranking Adaptation Latency per event.
    """

    def __init__(
        self,
        model:             lgb.Booster,
        feature_cols:      list,
        cfg:               dict,
        refit_window_size: int   = 200,   # how many recent rows to refit on
        recovery_threshold: float = 0.05, # within 5% of baseline = recovered
        window_size:       int   = 200,
        step_size:         int   = 50,
    ):
        self.model              = model
        self.feature_cols       = feature_cols
        self.cfg                = cfg
        self.refit_window_size  = refit_window_size
        self.recovery_threshold = recovery_threshold
        self.window_size        = window_size
        self.step_size          = step_size
        self.adaptation_results = []

    def run(
        self,
        df:           pd.DataFrame,
        drift_events: list,
        baseline_ndcg: float,
        scenario:     str
    ) -> list:
        """
        For each drift event:
          1. Refit model on rows just before the drift window
          2. Slide forward measuring NDCG@10 per window
          3. Count how many interactions until NDCG recovers within
             recovery_threshold of baseline
        """
        n = len(df)

        for event in drift_events:
            drift_start = event.window_start
            drift_end   = event.window_end
            ndcg_before = event.ndcg10_before
            ndcg_at_drift = event.ndcg10_current

            print(f"\nAdapting after drift at rows {drift_start}-{drift_end} "
                  f"(NDCG dropped {event.ndcg10_drop:.4f})")

            # Refit on the window immediately preceding the drift
            refit_start = max(0, drift_start - self.refit_window_size)
            refit_df    = df.iloc[refit_start:drift_start].copy()

            if len(refit_df) < 10:
                print("  Not enough rows to refit — skipping this event")
                continue

            # Only refit if we have multiple students
            if refit_df['id_student'].nunique() < 2:
                print("  Not enough unique students to refit — skipping")
                continue

            adapted_model = refit_model(
                self.model, refit_df, self.feature_cols, self.cfg
            )

            # Now slide forward from drift point, measuring recovery
            interactions_seen = 0
            recovered         = False
            ndcg_after        = ndcg_at_drift

            for start in range(drift_end, n - self.window_size + 1, self.step_size):
                end       = start + self.window_size
                window_df = df.iloc[start:end].copy()

                current_ndcg = compute_ndcg10(
                    adapted_model, window_df, self.feature_cols
                )
                interactions_seen += self.step_size
                ndcg_after         = current_ndcg

                recovery_gap = abs(current_ndcg - baseline_ndcg) / (baseline_ndcg + 1e-9)

                print(f"  +{interactions_seen} interactions: "
                      f"NDCG@10={current_ndcg:.4f} "
                      f"(gap={recovery_gap*100:.1f}%)")

                if recovery_gap <= self.recovery_threshold:
                    recovered = True
                    print(f"  [OK] Recovered in {interactions_seen} interactions")
                    break

            if not recovered:
                print(f"  [FAIL] Did not recover within {interactions_seen} interactions")

            result = AdaptationResult(
                scenario            = scenario,
                drift_window_start  = drift_start,
                drift_window_end    = drift_end,
                ndcg10_before       = round(ndcg_before, 4),
                ndcg10_after        = round(ndcg_after, 4),
                ndcg10_delta        = round(ndcg_after - ndcg_at_drift, 4),
                ndcg10_baseline     = round(baseline_ndcg, 4),
                interactions_to_recover = interactions_seen,
                refit_window_size   = self.refit_window_size,
                recovered           = recovered
            )
            self.adaptation_results.append(result)
            print(f"  AdaptationResult: latency={interactions_seen}, "
                  f"recovered={recovered}")

        return self.adaptation_results