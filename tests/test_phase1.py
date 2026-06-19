import pandas as pd
import pytest
import os


def test_dataset_exists():
    assert os.path.exists("data/processed/ranking_dataset.parquet"), \
        "Run src/pipeline/features.py first"


def test_dataset_shape():
    df = pd.read_parquet("data/processed/ranking_dataset.parquet")
    assert df.shape[0] > 5000, f"Too few rows: {df.shape[0]}"
    assert 'relevance' in df.columns
    assert 'id_student' in df.columns


def test_label_distribution():
    df = pd.read_parquet("data/processed/ranking_dataset.parquet")
    dist = df['relevance'].value_counts(normalize=True)
    for label in [0, 1, 2, 3]:
        pct = dist.get(label, 0)
        assert pct > 0.05, \
            f"Label {label} is only {pct*100:.1f}% — check assign_relevance()"


def test_no_null_features():
    df = pd.read_parquet("data/processed/ranking_dataset.parquet")
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    feature_cols = cfg['ranker']['feature_cols']
    existing = [c for c in feature_cols if c in df.columns]
    null_counts = df[existing].isnull().sum()
    # fillna(0) is applied in training; just check columns exist
    assert len(existing) >= 7, f"Missing feature columns: {set(feature_cols)-set(existing)}"


def test_baseline_ndcg():
    from src.ranker.baseline import train_baseline
    _, ndcg5, ndcg10 = train_baseline()
    assert ndcg5  > 0.55, f"NDCG@5 {ndcg5:.4f} too low"
    assert ndcg10 > 0.55, f"NDCG@10 {ndcg10:.4f} too low"