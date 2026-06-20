# src/monitoring/drift_simulator.py

import numpy as np
import pandas as pd
import yaml
import os


def simulate_skill_drift(
    df: pd.DataFrame,
    drift_start_frac: float = 0.5,
    drift_speed:      str   = "sudden",   # "sudden", "gradual", "slow"
    seed:             int   = 42
) -> pd.DataFrame:
    """
    Injects a synthetic skill-level shift into the dataset to simulate
    learner population drift for controlled benchmarking.

    Drift is applied to the feature space only — relevance labels are NOT
    changed, so NDCG will genuinely drop when the model sees shifted features.

    drift_speed:
      'sudden'  — all drift applied at drift_start_frac instantly
      'gradual' — linearly increasing drift from drift_start_frac to end
      'slow'    — same as gradual but half the magnitude per step
    """
    rng      = np.random.default_rng(seed)
    df_sim   = df.copy().reset_index(drop=True)
    n        = len(df_sim)
    drift_at = int(n * drift_start_frac)

    # Features that represent "engagement intensity" — these shift on skill change
    engagement_features = [
        'sum_click', 'n_active_weeks', 'avg_click_per_week', 'days_active_span'
    ]
    existing = [c for c in engagement_features if c in df_sim.columns]

    if drift_speed == "sudden":
        # Abrupt: all rows after drift_at get a large upward shift
        # (simulate learners suddenly engaging with harder courses)
        for col in existing:
            col_std = df_sim[col].std() + 1e-9
            noise   = rng.normal(loc=2.5 * col_std, scale=0.3 * col_std,
                                 size=n - drift_at)
            df_sim.loc[drift_at:, col] = (
                df_sim.loc[drift_at:, col].values + noise
            ).clip(min=0)

    elif drift_speed in ("gradual", "slow"):
        scale = 1.0 if drift_speed == "gradual" else 0.5
        for col in existing:
            col_std = df_sim[col].std() + 1e-9
            for i in range(drift_at, n):
                progress = (i - drift_at) / max(n - drift_at, 1)
                shift    = scale * progress * 2.0 * col_std
                noise    = rng.normal(loc=shift, scale=0.2 * col_std)
                df_sim.loc[i, col] = max(0, df_sim.loc[i, col] + noise)

    df_sim['drift_injected']   = False
    df_sim['drift_speed']      = drift_speed
    df_sim.loc[drift_at:, 'drift_injected'] = True

    print(f"Drift simulation: speed='{drift_speed}', "
          f"drift_at={drift_at}/{n} rows ({drift_start_frac*100:.0f}%), "
          f"affected_features={existing}")
    return df_sim


def generate_all_scenarios(config_path="config.yaml"):
    """Generate and save sudden, gradual, and slow drift scenarios."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg['data']['processed_path']
    df = pd.read_parquet(os.path.join(processed_path, 'ranking_dataset.parquet'))

    # Shuffle to simulate interaction stream order
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    scenarios = {}
    for speed in ["sudden", "gradual", "slow"]:
        sim_df = simulate_skill_drift(df, drift_speed=speed)
        out    = os.path.join(processed_path, f'sim_{speed}.parquet')
        sim_df.to_parquet(out, index=False)
        scenarios[speed] = sim_df
        print(f"Saved {speed} scenario → {out}")

    return scenarios


if __name__ == "__main__":
    generate_all_scenarios()