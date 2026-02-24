from llm_agents.llm_client import call_llm

def explanation_agent(validated):
    prompt = f"""
You are a medical explanation assistant.

The diagnosed eye condition is:
"{validated}"

Your task:
- Explain ONLY the above condition.
- If the condition is "Normal Retina", explain what a normal retina means.
- If the condition is "Cataract", explain cataract.
- If the condition is "Diabetic Retinopathy", explain diabetic retinopathy.
- If multiple conditions are mentioned, explain each briefly.

STRICT RULES:
- Do NOT assume any other disease.
- Do NOT explain Normal Retina unless it is explicitly mentioned.
- Do NOT add treatment or medication.
- Use simple, patient-friendly medical language.
- Stay strictly aligned with the given diagnosis.
"""

    return call_llm(prompt)