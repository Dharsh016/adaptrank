# tests/test_phase7.py

import os
import pytest


def test_dvc_initialized():
    assert os.path.exists(".dvc"), "Run: dvc init"
    assert os.path.exists(".dvcignore")


def test_dvc_pointer_files_exist():
    for fname in [
        "data/processed/ranking_dataset.parquet.dvc",
        "data/processed/baseline_model.txt.dvc",
    ]:
        assert os.path.exists(fname), \
            f"Missing DVC pointer: {fname} — run: dvc add {fname.replace('.dvc','')}"


def test_dvc_yaml_exists():
    assert os.path.exists("dvc.yaml"), "Missing dvc.yaml pipeline file"


def test_github_actions_workflow_exists():
    assert os.path.exists(".github/workflows/ci.yml"), \
        "Missing GitHub Actions workflow"


def test_prometheus_yml_exists():
    assert os.path.exists("prometheus.yml"), "Missing prometheus.yml"


def test_docker_compose_has_prometheus():
    with open("docker-compose.yml") as f:
        content = f.read()
    assert "prometheus" in content, "docker-compose.yml missing Prometheus service"
    assert "grafana"    in content, "docker-compose.yml missing Grafana service"


def test_metrics_endpoint_importable():
    """Verify Prometheus instrumentator is installed."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        assert Instrumentator is not None
    except ImportError:
        pytest.fail("prometheus-fastapi-instrumentator not installed — run: pip install prometheus-fastapi-instrumentator")


def test_app_has_metrics_route():
    """Verify /metrics endpoint exists in the FastAPI app."""
    from src.serving.app import app, load_resources
    load_resources()
    routes = [r.path for r in app.routes]
    assert "/metrics" in routes, \
        f"/metrics not found in routes: {routes}"