import os
import pickle
import yaml
import pandas as pd
import numpy as np


def build_learner_profiles(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg['data']['processed_path']

    df = pd.read_parquet(os.path.join(processed_path, 'ranking_dataset.parquet'))

    emb_path = os.path.join(processed_path, 'course_embeddings.pkl')
    with open(emb_path, 'rb') as f:
        course_emb_map = pickle.load(f)

    emb_dim = next(iter(course_emb_map.values())).shape[0]

    profiles = {}
    missing_modules = set()

    for sid, grp in df.groupby('id_student'):
        vecs, weights = [], []
        for _, row in grp.iterrows():
            emb = course_emb_map.get(row['code_module'])
            if emb is not None:
                vecs.append(emb)
                # relevance + 1 so label-0 rows still contribute
                weights.append(float(row['relevance']) + 1.0)
            else:
                missing_modules.add(row['code_module'])

        if vecs:
            vecs    = np.array(vecs, dtype=np.float32)
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            profiles[sid] = np.average(vecs, axis=0, weights=weights)
        else:
            # Fallback: zero vector — student has no embeddable courses
            profiles[sid] = np.zeros(emb_dim, dtype=np.float32)

    if missing_modules:
        print(f"Warning: {len(missing_modules)} modules had no embedding: {missing_modules}")

    profile_path = os.path.join(processed_path, 'learner_profiles.pkl')
    with open(profile_path, 'wb') as f:
        pickle.dump(profiles, f)

    print(f"Built {len(profiles)} learner profiles, dim={emb_dim}")
    print(f"Profile path: {profile_path}")
    return profiles


if __name__ == "__main__":
    build_learner_profiles()