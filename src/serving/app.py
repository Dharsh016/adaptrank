# src/serving/app.py

import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.serving.schemas import (
    RankRequest, RankResponse,
    AdaptRequest, AdaptResponse,
    HealthResponse, StudentsResponse
)

# src/serving/app.py — add after imports, before app definition

from prometheus_client import Counter, Histogram, Gauge
import time

# Custom metrics
RANK_REQUESTS      = Counter("adaptrank_rank_requests_total",
                             "Total /rank requests", ["status"])
ADAPT_REQUESTS     = Counter("adaptrank_adapt_requests_total",
                             "Total /adapt requests", ["scenario", "status"])
RANK_LATENCY       = Histogram("adaptrank_rank_latency_seconds",
                               "Response time for /rank")
ADAPT_LATENCY      = Histogram("adaptrank_adapt_latency_seconds",
                               "Response time for /adapt")
CURRENT_NDCG       = Gauge("adaptrank_current_ndcg",
                           "Most recent NDCG@10 from /rank calls")
DRIFT_EVENTS_TOTAL = Gauge("adaptrank_drift_events_detected",
                           "Total drift events in last /adapt call", ["scenario"])

# ------------------------------------------------------------------ #
# Global state — loaded once at startup
# ------------------------------------------------------------------ #
MODEL        = None
DF           = None
FEATURE_COLS = None
CFG          = None
COURSE_EMB   = None
PROFILES     = None


def load_resources(config_path: str = "config.yaml"):
    global MODEL, DF, FEATURE_COLS, CFG, COURSE_EMB, PROFILES

    with open(config_path) as f:
        CFG = yaml.safe_load(f)

    processed_path = CFG["data"]["processed_path"]
    FEATURE_COLS   = CFG["ranker"]["feature_cols"]

    MODEL = lgb.Booster(
        model_file=os.path.join(processed_path, "baseline_model.txt")
    )
    DF = pd.read_parquet(
        os.path.join(processed_path, "ranking_dataset.parquet")
    )

    emb_path = os.path.join(processed_path, "course_embeddings.pkl")
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            COURSE_EMB = pickle.load(f)

    prof_path = os.path.join(processed_path, "learner_profiles.pkl")
    if os.path.exists(prof_path):
        with open(prof_path, "rb") as f:
            PROFILES = pickle.load(f)

    print(f"Resources loaded: model={MODEL is not None}, "
          f"df={DF.shape}, profiles={len(PROFILES) if PROFILES else 0}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield


app = FastAPI(
    title       = "AdaptRank API",
    description = "Learning-to-rank with drift-aware adaptation for OULAD",
    version     = "1.0.0",
    lifespan    = lifespan
)
from prometheus_fastapi_instrumentator import Instrumentator
# Instrument the API for Prometheus metrics
Instrumentator().instrument(app).expose(app)

# ------------------------------------------------------------------ #
# Health check
# ------------------------------------------------------------------ #
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status       = "ok",
        model_loaded = MODEL is not None
    )


# ------------------------------------------------------------------ #
# /students — List available students
# ------------------------------------------------------------------ #
@app.get("/students", response_model=StudentsResponse)
def get_students():
    """
    Returns list of all available student IDs in the dataset.
    Useful for UI population and validation.
    """
    if DF is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    student_ids = sorted(DF["id_student"].unique().tolist())
    return StudentsResponse(
        total_students = len(student_ids),
        student_ids    = student_ids,
        message        = f"Found {len(student_ids)} students in dataset"
    )


# ------------------------------------------------------------------ #
# /rank — Dev A
# ------------------------------------------------------------------ #
@app.post("/rank", response_model=RankResponse)
def rank_courses(request: RankRequest):
    start = time.time()
    try:
        if MODEL is None or DF is None:
            RANK_REQUESTS.labels(status="error").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        student_df = DF[DF["id_student"] == request.student_id].copy()
        if student_df.empty:
            RANK_REQUESTS.labels(status="not_found").inc()
            raise HTTPException(status_code=404,
                detail=f"Student {request.student_id} not found")

        X      = student_df[FEATURE_COLS].fillna(0).values
        scores = MODEL.predict(X)
        order  = np.argsort(scores)[::-1][:request.top_k]
        courses = student_df["code_module"].values[order].tolist()
        s_list  = [round(float(s), 4) for s in scores[order]]

        ndcg_val = None
        if "relevance" in student_df.columns and len(student_df) >= 2:
            from sklearn.metrics import ndcg_score as sk_ndcg
            y = student_df["relevance"].values
            try:
                ndcg_val = round(float(
                    sk_ndcg(y.reshape(1,-1), scores.reshape(1,-1), k=10)
                ), 4)
                CURRENT_NDCG.set(ndcg_val)
            except Exception:
                pass

        RANK_REQUESTS.labels(status="ok").inc()
        RANK_LATENCY.observe(time.time() - start)

        return RankResponse(
            student_id=request.student_id,
            ranked_courses=courses,
            scores=s_list,
            ndcg_current=ndcg_val,
            message=f"Ranked {len(courses)} courses"
        )
    except HTTPException:
        raise
    except Exception as e:
        RANK_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------ #
# /adapt — Dev B
# ------------------------------------------------------------------ #
@app.post("/adapt", response_model=AdaptResponse)
def adapt_model(request: AdaptRequest):
    start = time.time()
    try:
        if MODEL is None:
            ADAPT_REQUESTS.labels(scenario=request.scenario, status="error").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        valid_scenarios = ["sudden", "gradual", "slow"]
        if request.scenario not in valid_scenarios:
            ADAPT_REQUESTS.labels(scenario="invalid", status="error").inc()
            raise HTTPException(status_code=400,
                detail=f"scenario must be one of {valid_scenarios}")

        sim_path = os.path.join(
            CFG["data"]["processed_path"], f"sim_{request.scenario}.parquet"
        )
        if not os.path.exists(sim_path):
            raise HTTPException(status_code=404,
                detail=f"Simulation file not found: {sim_path}")

        from src.agents.pipeline import run_pipeline
        final_state = run_pipeline(request.scenario)

        n_events = len(final_state.get("drift_events") or [])
        DRIFT_EVENTS_TOTAL.labels(scenario=request.scenario).set(n_events)
        ADAPT_REQUESTS.labels(scenario=request.scenario, status="ok").inc()
        ADAPT_LATENCY.observe(time.time() - start)

        return AdaptResponse(
            scenario=request.scenario,
            n_drift_events=n_events,
            avg_latency=round(final_state.get("avg_latency") or 0.0, 2),
            recovery_rate=round(final_state.get("recovery_rate") or 0.0, 4),
            baseline_ndcg=round(final_state.get("baseline_ndcg") or 0.0, 4),
            explanation=final_state.get("explanation_report") or "N/A",
            message="Adaptation complete"
        )
    except HTTPException:
        raise
    except Exception as e:
        ADAPT_REQUESTS.labels(scenario=request.scenario, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))