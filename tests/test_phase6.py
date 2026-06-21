# tests/test_phase6.py

import os
import pytest
import pandas as pd


# ------------------------------------------------------------------ #
# Great Expectations tests
# ------------------------------------------------------------------ #
def test_validation_passes():
    import yaml
    from src.pipeline.validation import validate_ranking_dataset
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    result = validate_ranking_dataset(
        os.path.join(cfg["data"]["processed_path"], "ranking_dataset.parquet")
    )
    assert result["success"] == True, \
        f"Validation failed on: {result['failed_checks']}"


def test_validation_catches_bad_data():
    """Confirm validator catches obviously wrong data."""
    import tempfile
    import numpy as np
    from src.pipeline.validation import validate_ranking_dataset

    bad_df = pd.DataFrame({
        "id_student":   [None, None],
        "code_module":  ["AAA", "BBB"],
        "relevance":    [5, 99],   # out of range
        "sum_click":    [-1, -2],  # negative
        "n_active_weeks": [1, 1]
    })
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        bad_df.to_parquet(tmp.name)
        tmp_path = tmp.name

    result = validate_ranking_dataset(tmp_path)
    os.unlink(tmp_path)
    assert result["success"] == False
    assert len(result["failed_checks"]) > 0


# ------------------------------------------------------------------ #
# FastAPI endpoint tests — use TestClient (no server needed)
# ------------------------------------------------------------------ #
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.serving.app import app, load_resources
    load_resources()
    return TestClient(app)


@pytest.fixture(scope="module")
def valid_student_id():
    df = pd.read_parquet("data/processed/ranking_dataset.parquet")
    return int(df["id_student"].iloc[0])


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == True


def test_rank_endpoint_valid_student(client, valid_student_id):
    response = client.post(
        "/rank",
        json={"student_id": valid_student_id, "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"]     == valid_student_id
    assert len(data["ranked_courses"]) <= 5
    assert len(data["scores"])         == len(data["ranked_courses"])
    assert all(isinstance(s, float) for s in data["scores"])


def test_rank_endpoint_invalid_student(client):
    response = client.post(
        "/rank",
        json={"student_id": 9999999, "top_k": 5}
    )
    assert response.status_code == 404


def test_adapt_endpoint_valid_scenario(client):
    response = client.post(
        "/adapt",
        json={"scenario": "sudden", "refit_window_size": 200}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scenario"]       == "sudden"
    assert data["n_drift_events"]  > 0
    assert 0 <= data["recovery_rate"] <= 1
    assert len(data["explanation"]) > 20


def test_adapt_endpoint_invalid_scenario(client):
    response = client.post(
        "/adapt",
        json={"scenario": "nonexistent"}
    )
    assert response.status_code == 400


def test_schemas_importable():
    from src.serving.schemas import (
        RankRequest, RankResponse,
        AdaptRequest, AdaptResponse,
        HealthResponse
    )
    r = RankRequest(student_id=123, top_k=5)
    assert r.student_id == 123
    assert r.top_k      == 5