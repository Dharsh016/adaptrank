import os
import pickle
import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def build_course_embeddings(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_path       = cfg['data']['raw_path']
    processed_path = cfg['data']['processed_path']
    cache_path     = os.path.join(processed_path, 'course_embeddings.pkl')

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    courses = pd.read_csv(os.path.join(raw_path, 'courses.csv'))

    # Build text per course — code_module is the human-readable course ID
    if 'module_presentation_length' in courses.columns:
        courses['text'] = (
            "University course module: " + courses['code_module'].astype(str) +
            ". Presentation: " + courses['code_presentation'].astype(str) +
            ". Duration: " + courses['module_presentation_length'].astype(str) + " days."
        )
    else:
        courses['text'] = (
            "University course module: " + courses['code_module'].astype(str) +
            ". Presentation: " + courses['code_presentation'].astype(str) + "."
        )

    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Embedding {len(courses)} course records...")
    embeddings = model.encode(
        courses['text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    # Key by code_module (not code_presentation) for learner profile lookup
    course_emb_map = {}
    for i, row in courses.iterrows():
        key = row['code_module']
        if key not in course_emb_map:
            course_emb_map[key] = embeddings[i]
        else:
            # Average across presentations of same module
            course_emb_map[key] = (course_emb_map[key] + embeddings[i]) / 2.0

    with open(cache_path, 'wb') as f:
        pickle.dump(course_emb_map, f)

    print(f"Saved {len(course_emb_map)} module embeddings → {cache_path}")
    print(f"Embedding dim: {next(iter(course_emb_map.values())).shape[0]}")
    return course_emb_map


if __name__ == "__main__":
    build_course_embeddings()