import os
import pickle
import pytest
import yaml


def test_course_embeddings_exist():
    assert os.path.exists("data/processed/course_embeddings.pkl")


def test_course_embeddings_count():
    with open("data/processed/course_embeddings.pkl", "rb") as f:
        emb = pickle.load(f)
    assert len(emb) >= 7, f"Expected >=7 unique modules, got {len(emb)}"
    first = next(iter(emb.values()))
    assert first.shape[0] == 384, f"Expected 384-dim embeddings, got {first.shape[0]}"


def test_learner_profiles_exist():
    assert os.path.exists("data/processed/learner_profiles.pkl")


def test_learner_profiles_count():
    with open("data/processed/learner_profiles.pkl", "rb") as f:
        profiles = pickle.load(f)
    assert len(profiles) > 3000, f"Expected >3k profiles, got {len(profiles)}"


def test_optuna_params_in_config():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    for key in ['num_leaves', 'learning_rate', 'lambda_l2']:
        assert key in cfg['ranker'], f"Missing {key} in config.yaml ranker section"


def test_tuned_ndcg_beats_baseline():
    from src.ranker.evaluate import full_cv_eval
    _, _, mean10 = full_cv_eval(n_splits=3)   # 3-fold for speed in test
    assert mean10 > 0.60, f"Tuned NDCG@10 {mean10:.4f} did not beat baseline"