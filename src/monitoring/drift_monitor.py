# src/monitoring/drift_monitor.py

import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass, asdict
from datetime import datetime
from sklearn.metrics import ndcg_score
from typing import Optional


@dataclass
class DriftEvent:
    """Structured record of a single detected drift event."""
    timestamp:           str
    window_start:        int          # interaction index
    window_end:          int
    ndcg10_before:       float
    ndcg10_current:      float
    ndcg10_drop:         float
    threshold_used:      float
    drift_confirmed:     bool
    top_shifted_features: list        # feature names with largest mean-shift
    student_cohort_size: int


class SlidingWindowNDCGMonitor:
    """
    Monitors ranking quality over a sliding window of student interactions.
    Triggers a DriftEvent when NDCG@10 drops by more than `threshold` below
    the rolling baseline established in the warm-up period.
    """

    def __init__(
        self,
        model:          lgb.Booster,
        feature_cols:   list,
        window_size:    int   = 200,    # interactions per window
        step_size:      int   = 50,     # slide step
        threshold:      float = 0.08,   # NDCG drop that triggers drift alert
        warmup_windows: int   = 3,      # windows used to establish baseline
    ):
        self.model           = model
        self.feature_cols    = feature_cols
        self.window_size     = window_size
        self.step_size       = step_size
        self.threshold       = threshold
        self.warmup_windows  = warmup_windows
        self.baseline_ndcg   = None
        self.window_ndcg_log = []       # [(window_idx, ndcg10)]
        self.drift_events    = []

    def _compute_window_ndcg10(self, window_df: pd.DataFrame) -> float:
        """Compute mean NDCG@10 across all students in this window."""
        X     = window_df[self.feature_cols].fillna(0).values
        preds = self.model.predict(X)
        y     = window_df['relevance'].values
        sids  = window_df['id_student'].values

        scores = []
        for sid in np.unique(sids):
            mask = sids == sid
            if mask.sum() < 2:
                continue
            scores.append(
                ndcg_score(y[mask].reshape(1,-1),
                           preds[mask].reshape(1,-1), k=10)
            )
        return float(np.mean(scores)) if scores else 0.0

    def _top_shifted_features(
        self,
        before_df: pd.DataFrame,
        after_df:  pd.DataFrame,
        top_n:     int = 3
    ) -> list:
        """
        Return names of features whose mean value shifted most between
        before_df and after_df. Used for lightweight root-cause attribution
        without SHAP (fast, always works).
        """
        shifts = {}
        for col in self.feature_cols:
            if col in before_df.columns and col in after_df.columns:
                b_mean = before_df[col].fillna(0).mean()
                a_mean = after_df[col].fillna(0).mean()
                denom  = abs(b_mean) + 1e-9
                shifts[col] = abs(a_mean - b_mean) / denom
        return sorted(shifts, key=shifts.get, reverse=True)[:top_n]

    def run(self, df: pd.DataFrame) -> list:
        """
        Slide window across df rows. df must be sorted by interaction order
        (e.g. by a simulated time index). Returns list of DriftEvent objects.
        """
        n          = len(df)
        window_idx = 0
        prev_window_df = None

        for start in range(0, n - self.window_size + 1, self.step_size):
            end        = start + self.window_size
            window_df  = df.iloc[start:end].copy()
            ndcg10     = self._compute_window_ndcg10(window_df)

            self.window_ndcg_log.append((window_idx, start, end, ndcg10))

            # Establish baseline from first warmup_windows windows
            if window_idx < self.warmup_windows:
                window_idx += 1
                prev_window_df = window_df
                continue

            if self.baseline_ndcg is None:
                warmup_scores    = [v[3] for v in self.window_ndcg_log[:self.warmup_windows]]
                self.baseline_ndcg = float(np.mean(warmup_scores))
                print(f"Baseline NDCG@10 established: {self.baseline_ndcg:.4f}")

            drop = self.baseline_ndcg - ndcg10

            if drop >= self.threshold:
                shifted = (
                    self._top_shifted_features(prev_window_df, window_df)
                    if prev_window_df is not None else []
                )
                event = DriftEvent(
                    timestamp            = datetime.utcnow().isoformat(),
                    window_start         = int(start),
                    window_end           = int(end),
                    ndcg10_before        = round(self.baseline_ndcg, 4),
                    ndcg10_current       = round(ndcg10, 4),
                    ndcg10_drop          = round(drop, 4),
                    threshold_used       = self.threshold,
                    drift_confirmed      = True,
                    top_shifted_features = shifted,
                    student_cohort_size  = int(window_df['id_student'].nunique())
                )
                self.drift_events.append(event)
                print(f"[DRIFT DETECTED] window={window_idx} "
                      f"NDCG@10={ndcg10:.4f} (drop={drop:.4f}) "
                      f"shifted_features={shifted}")
            else:
                # Soft update: rolling baseline drifts slowly upward if improving
                self.baseline_ndcg = 0.9 * self.baseline_ndcg + 0.1 * ndcg10

            prev_window_df = window_df
            window_idx += 1

        print(f"\nMonitor complete. {len(self.drift_events)} drift events detected "
              f"across {window_idx} windows.")
        return self.drift_events

    def save_log(self, out_path: str):
        """Save full window NDCG log and drift events to JSON."""
        log = {
            "window_log": [
                {"window_idx": w[0], "start": w[1], "end": w[2], "ndcg10": round(w[3],4)}
                for w in self.window_ndcg_log
            ],
            "drift_events": [asdict(e) for e in self.drift_events],
            "baseline_ndcg": self.baseline_ndcg,
            "threshold": self.threshold,
        }
        with open(out_path, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"Drift log saved → {out_path}")