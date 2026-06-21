# src/pipeline/validation.py

import os
import pandas as pd


def validate_ranking_dataset(parquet_path: str) -> dict:
    """
    Validates the ranking dataset against defined expectations.
    Called at pipeline start before any ML work.
    Returns dict with success flag and any failed expectations.
    """
    df = pd.read_parquet(parquet_path)
    
    failures = []
    checks_performed = 0

    # Schema expectations
    required_cols = ["id_student", "code_module", "relevance", "sum_click", "n_active_weeks"]
    for col in required_cols:
        checks_performed += 1
        if col not in df.columns:
            failures.append(f"missing_column_{col}")

    # Value range expectations
    if "relevance" in df.columns:
        checks_performed += 1
        if not df["relevance"].between(0, 3, inclusive="both").all():
            failures.append("relevance_out_of_range")

    if "sum_click" in df.columns:
        checks_performed += 1
        if (df["sum_click"] < 0).any():
            failures.append("sum_click_negative")

    # Null value checks
    if "id_student" in df.columns:
        checks_performed += 1
        if df["id_student"].isnull().any():
            failures.append("id_student_has_nulls")

    if "relevance" in df.columns:
        checks_performed += 1
        if df["relevance"].isnull().any():
            failures.append("relevance_has_nulls")

    # Row count
    checks_performed += 1
    if not (5000 <= len(df) <= 200000):
        failures.append(f"row_count_out_of_range_{len(df)}")

    success = len(failures) == 0

    print(f"Validation: {'PASSED' if success else 'FAILED'} ({checks_performed} checks)")
    if failures:
        print(f"  Failed checks: {failures}")

    return {
        "success":          success,
        "failed_checks":    failures,
        "n_checked":        checks_performed,
        "evaluated_at":     "validation_complete"
    }


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    result = validate_ranking_dataset(
        os.path.join(cfg["data"]["processed_path"], "ranking_dataset.parquet")
    )
    print(result)