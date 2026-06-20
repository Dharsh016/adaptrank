import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` can be imported
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.pipeline.embeddings import build_course_embeddings
from src.pipeline.profiles import build_learner_profiles


if __name__ == '__main__':
    print('Building course embeddings...')
    build_course_embeddings()
    print('Building learner profiles...')
    build_learner_profiles()
    print('Done')
