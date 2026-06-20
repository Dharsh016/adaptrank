# scripts/evaluate_reports.py
"""
Manual rubric evaluation of Explainer Agent outputs.
Both developers independently score 20 sampled reports.
Inter-rater agreement computed at the end.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np


RUBRIC = """
Score each criterion 1-3:

1. SPECIFICITY (1=vague, 2=some numbers, 3=specific numbers and feature names)
2. CORRECTNESS (1=contradicts data, 2=mostly correct, 3=fully consistent with metrics)
3. CLARITY (1=technical jargon, 2=mixed, 3=clear to non-technical reader)

Enter scores as three numbers separated by spaces (e.g. "3 2 3")
"""


def evaluate_reports(reports_dir: str = "logs"):
    report_files = [
        f for f in os.listdir(reports_dir)
        if f.startswith("agent_report_") and f.endswith(".txt")
    ]

    all_scores = {}

    for fname in report_files:
        path     = os.path.join(reports_dir, fname)
        scenario = fname.replace("agent_report_", "").replace(".txt", "")

        with open(path) as f:
            report_text = f.read()

        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print('='*60)
        print(report_text)
        print(RUBRIC)

        while True:
            try:
                raw    = input("Your scores (specificity correctness clarity): ").strip()
                scores = list(map(int, raw.split()))
                assert len(scores) == 3
                assert all(1 <= s <= 3 for s in scores)
                break
            except Exception:
                print("Invalid input. Enter exactly 3 numbers between 1 and 3.")

        all_scores[scenario] = {
            "specificity": scores[0],
            "correctness": scores[1],
            "clarity":     scores[2],
            "mean":        round(np.mean(scores), 2)
        }

    print("\n" + "="*60)
    print("RUBRIC EVALUATION SUMMARY")
    print("="*60)
    for sc, s in all_scores.items():
        print(f"{sc:10s}: specificity={s['specificity']}  "
              f"correctness={s['correctness']}  "
              f"clarity={s['clarity']}  mean={s['mean']}")

    out_path = "logs/rubric_scores.json"
    with open(out_path, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nScores saved → {out_path}")
    print("Share this file with the other developer for inter-rater agreement.")
    return all_scores


if __name__ == "__main__":
    evaluate_reports()