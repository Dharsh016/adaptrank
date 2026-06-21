# src/serving/schemas.py

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class RankRequest(BaseModel):
    student_id: int
    top_k: int = 10


class RankResponse(BaseModel):
    student_id:     int
    ranked_courses: List[str]
    scores:         List[float]
    ndcg_current:   Optional[float] = None
    message:        str = "ok"


class AdaptRequest(BaseModel):
    scenario:          str = "sudden"   # sudden, gradual, slow
    refit_window_size: int = 200


class AdaptResponse(BaseModel):
    scenario:          str
    n_drift_events:    int
    avg_latency:       float
    recovery_rate:     float
    baseline_ndcg:     float
    explanation:       str
    message:           str = "ok"


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    version:     str = "1.0.0"


class StudentsResponse(BaseModel):
    total_students: int
    student_ids:    List[int]
    message:        str = "ok"