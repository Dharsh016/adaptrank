# src/agents/explainer_agent.py

import os
import litellm
from dotenv import load_dotenv
from src.agents.state import AdaptRankState

load_dotenv()
litellm.set_verbose = False

# Groq free tier — LLaMA-3.1-8B, runs in cloud, ~1-2 sec response
# No local model, no RAM, no GPU needed
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "groq/llama-3.1-8b-instant"
LLM_BASE     = None   # Groq uses its own endpoint, no base_url needed


EXPLAINER_SYSTEM_PROMPT = """You are an ML monitoring expert specializing in 
learning-to-rank systems. You receive structured data about a ranking quality 
drift event and the system's adaptation response. 

Your job is to write a clear, concise adaptation report in 3 short paragraphs:
1. What drifted and which features were most responsible
2. How the model adapted and how fast it recovered  
3. What this means for the learner experience

Be specific with numbers. Do not use bullet points. Write in plain English 
that a product manager could understand. Maximum 200 words total."""


def build_explainer_prompt(state: AdaptRankState) -> str:
    drift_events    = state.get('drift_events', [])
    profile_summary = state.get('profile_shift_summary', 'N/A')
    avg_latency     = state.get('avg_latency', 0)
    recovery_rate   = state.get('recovery_rate', 0)
    baseline_ndcg   = state.get('baseline_ndcg', 0)
    scenario        = state.get('scenario', 'unknown')
    n_events        = len(drift_events)

    first_ndcg_drop = drift_events[0]['ndcg10_drop'] if drift_events else 0
    top_features    = drift_events[0].get('top_shifted_features', []) if drift_events else []

    return f"""
Drift scenario: {scenario}
Total drift events detected: {n_events}
Baseline NDCG@10: {baseline_ndcg:.4f}
First event NDCG drop: {first_ndcg_drop:.4f}
Top shifted features: {', '.join(top_features) if top_features else 'N/A'}
Profile shift analysis: {profile_summary}
Average adaptation latency: {avg_latency:.1f} interactions
Recovery rate: {recovery_rate*100:.1f}% of events recovered within 5% of baseline
Refit window size used: {state.get('refit_window_used', 200)} interactions

Write the adaptation report now.
""".strip()


def explainer_agent_node(state: AdaptRankState) -> AdaptRankState:
    """
    Node 4: LLM generates plain-language adaptation report.
    Uses Groq free API (llama-3.1-8b-instant) via LiteLLM.
    Falls back to structured text if API unavailable.
    """
    print("[Explainer Agent] Generating adaptation report via Groq...")

    if not state.get('drift_detected'):
        return {
            **state,
            'explanation_report': "No drift detected in this scenario. Model ranking quality remained stable.",
            'rubric_scores': {}
        }

    prompt = build_explainer_prompt(state)

    report = None

    # Primary: Groq free API
    if GROQ_API_KEY:
        try:
            response = litellm.completion(
                model    = LLM_MODEL,
                messages = [
                    {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                api_key    = GROQ_API_KEY,
                max_tokens = 300,
                temperature = 0.3
            )
            report = response.choices[0].message.content.strip()
            print(f"[Explainer Agent] Groq response received ({len(report)} chars).")
        except Exception as e:
            print(f"[Explainer Agent] Groq call failed: {e}")

    # Fallback: structured text report (no LLM needed, always works)
    if not report:
        print("[Explainer Agent] Using structured fallback report.")
        drift_events = state.get('drift_events', [])
        top_features = drift_events[0].get('top_shifted_features', []) if drift_events else []
        report = (
            f"Drift Report — Scenario: {state['scenario'].upper()}\n\n"
            f"The ranking system detected {len(drift_events)} drift events against a "
            f"baseline NDCG@10 of {state.get('baseline_ndcg', 0):.4f}. "
            f"The most significant feature shifts were observed in: "
            f"{', '.join(top_features) if top_features else 'engagement features'}. "
            f"These shifts indicate a change in learner engagement patterns, "
            f"consistent with a '{state['scenario']}' drift profile.\n\n"
            f"The adaptation engine triggered LightGBM refit using a "
            f"{state.get('refit_window_used', 200)}-interaction window. "
            f"Average recovery latency was {state.get('avg_latency', 0):.1f} interactions, "
            f"with {state.get('recovery_rate', 0)*100:.1f}% of events recovering "
            f"within 5% of the baseline NDCG.\n\n"
            f"Profile shift analysis: {state.get('profile_shift_summary', 'N/A')} "
            f"Learners affected by this drift may have experienced temporarily degraded "
            f"course recommendations until the model adapted."
        )

    print(f"\n{'='*50}\nREPORT:\n{'='*50}\n{report}\n{'='*50}")

    return {
        **state,
        'explanation_report': report,
        'rubric_scores':      {}
    }