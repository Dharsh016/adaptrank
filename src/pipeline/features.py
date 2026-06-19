import pandas as pd
import numpy as np
import yaml
import os
from src.pipeline.labels import assign_relevance


def load_raw_data(raw_path):
    student_info = pd.read_csv(os.path.join(raw_path, 'studentInfo.csv'))
    student_reg  = pd.read_csv(os.path.join(raw_path, 'studentRegistration.csv'))
    student_vle  = pd.read_csv(os.path.join(raw_path, 'studentVle.csv'))
    courses      = pd.read_csv(os.path.join(raw_path, 'courses.csv'))
    return student_info, student_reg, student_vle, courses


def encode_categoricals(df):
    for col in ['age_band', 'highest_education', 'disability', 'gender', 'region']:
        if col in df.columns:
            df[col + '_enc'] = pd.factorize(df[col])[0]
    return df


def build_ranking_dataset(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_path       = cfg['data']['raw_path']
    processed_path = cfg['data']['processed_path']
    os.makedirs(processed_path, exist_ok=True)

    print("Loading raw OULAD files...")
    student_info, student_reg, student_vle, courses = load_raw_data(raw_path)

    # ------------------------------------------------------------------ #
    # 1. Aggregate VLE clicks per student-course
    # ------------------------------------------------------------------ #
    print("Aggregating VLE activity...")
    vle_agg = (
        student_vle
        .groupby(['id_student', 'code_module', 'code_presentation'])
        .agg(
            sum_click          = ('sum_click', 'sum'),
            n_active_weeks     = ('date', 'nunique'),
            avg_click_per_week = ('sum_click', 'mean'),
            first_activity_day = ('date', 'min'),
            last_activity_day  = ('date', 'max'),
        )
        .reset_index()
    )
    vle_agg['days_active_span'] = (
        vle_agg['last_activity_day'] - vle_agg['first_activity_day']
    )

    # ------------------------------------------------------------------ #
    # 2. Merge student info with VLE aggregates
    # ------------------------------------------------------------------ #
    print("Merging student info with VLE aggregates...")
    df = student_info.merge(
        vle_agg,
        on=['id_student', 'code_module', 'code_presentation'],
        how='left'
    )

    # Fill students who never accessed VLE
    for col in ['sum_click','n_active_weeks','avg_click_per_week',
                'days_active_span','first_activity_day','last_activity_day']:
        df[col] = df[col].fillna(0)

    # ------------------------------------------------------------------ #
    # 3. Merge unregistration dates
    # ------------------------------------------------------------------ #
    df = df.merge(
        student_reg[['id_student','code_module',
                     'code_presentation','date_unregistration']],
        on=['id_student','code_module','code_presentation'],
        how='left'
    )

    # ------------------------------------------------------------------ #
    # 4. Merge course duration
    # ------------------------------------------------------------------ #
    df = df.merge(courses, on=['code_module','code_presentation'], how='left')

    # ------------------------------------------------------------------ #
    # 5. Encode categoricals
    # ------------------------------------------------------------------ #
    df = encode_categoricals(df)

    # ------------------------------------------------------------------ #
    # 6. Compute per-course click 75th percentile for label threshold
    # ------------------------------------------------------------------ #
    click_75th_map = (
        df.groupby('code_module')['sum_click']
        .quantile(0.75)
        .to_dict()
    )
    df['click_75th'] = df['code_module'].map(click_75th_map)

    # ------------------------------------------------------------------ #
    # 7. Assign relevance labels
    # ------------------------------------------------------------------ #
    print("Assigning relevance labels...")
    df['relevance'] = df.apply(
        lambda r: assign_relevance(
            r['final_result'],
            r.get('date_unregistration', '?'),
            r['sum_click'],
            r['click_75th']
        ),
        axis=1
    )

    # ------------------------------------------------------------------ #
    # 8. Keep only students who have >= 2 courses (needed for NDCG)
    # ------------------------------------------------------------------ #
    course_counts = df.groupby('id_student')['code_module'].count()
    valid_students = course_counts[course_counts >= 2].index
    df = df[df['id_student'].isin(valid_students)].copy()

    # ------------------------------------------------------------------ #
    # 9. Validate label distribution
    # ------------------------------------------------------------------ #
    dist = df['relevance'].value_counts(normalize=True)
    print("\nLabel distribution:")
    for label in [0, 1, 2, 3]:
        pct = dist.get(label, 0) * 100
        flag = " ← CHECK THIS" if pct < 5 or pct > 65 else ""
        print(f"  label {label}: {pct:.1f}%{flag}")

    # ------------------------------------------------------------------ #
    # 10. Save
    # ------------------------------------------------------------------ #
    out_path = os.path.join(processed_path, 'ranking_dataset.parquet')
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {df.shape[0]} rows × {df.shape[1]} cols → {out_path}")
    print(f"Unique students: {df['id_student'].nunique()}")
    print(f"Unique courses:  {df['code_module'].nunique()}")
    return df


if __name__ == "__main__":
    build_ranking_dataset()