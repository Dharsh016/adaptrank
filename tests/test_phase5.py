# tests/test_phase5.py

import os
import json
import pytest


def test_pipeline_imports():
    from src.agents.pipeline import build_pipeline
    pipeline = build_pipeline()
    assert pipeline is not None


def test_state_schema():
    from src.agents.state import AdaptRankState
    import typing
    hints = typing.get_type_hints(AdaptRankState)
    for key in ['scenario', 'drift_events', 'explanation_report', 'adaptation_results']:
        assert key in hints, f"Missing key in state: {key}"


def test_agent_reports_exist():
    for scenario in ["sudden", "gradual", "slow"]:
        path = f"logs/agent_report_{scenario}.txt"
        assert os.path.exists(path), \
            f"Missing {path} — run scripts/run_phase5.py first"


def test_agent_reports_non_empty():
    for scenario in ["sudden", "gradual", "slow"]:
        path = f"logs/agent_report_{scenario}.txt"
        if os.path.exists(path):
            with open(path) as f:
                content = f.read().strip()
            assert len(content) > 50, \
                f"Report for {scenario} is too short: '{content}'"


def test_agent_state_json_exists():
    for scenario in ["sudden", "gradual", "slow"]:
        path = f"logs/agent_state_{scenario}.json"
        assert os.path.exists(path), \
            f"Missing {path} — run scripts/run_phase5.py first"


def test_agent_state_has_all_fields():
    path = "logs/agent_state_sudden.json"
    if os.path.exists(path):
        with open(path) as f:
            state = json.load(f)
        for key in ['drift_detected', 'avg_latency', 'recovery_rate',
                    'explanation_report', 'profile_shift_summary']:
            assert key in state, f"Missing field in agent state: {key}"
        assert state['drift_detected'] == True
        assert state['avg_latency'] is not None


def test_groq_reachable():
    """Checks Groq API key is set and reachable. Skip if key missing."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set in .env — using fallback report mode")
    try:
        import litellm
        r = litellm.completion(
            model    = "groq/llama-3.1-8b-instant",
            messages = [{"role": "user", "content": "Reply with READY only."}],
            api_key    = api_key,
            max_tokens = 10
        )
        reply = r.choices[0].message.content.strip()
        assert len(reply) > 0
        print(f"Groq response: {reply}")
    except Exception as e:
        pytest.skip(f"Groq not reachable: {e}")