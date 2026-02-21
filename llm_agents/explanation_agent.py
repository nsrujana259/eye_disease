from llm_agents.llm_client import call_llm

def explanation_agent(validated):
    prompt = f"""
Explain the eye diagnosis in simple, medical, and patient-friendly terms.

Diagnosis:
{validated}

Rules:
- No treatment or medication
- Clear explanation
"""
    return call_llm(prompt)