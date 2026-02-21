from llm_agents.llm_client import call_llm

def report_agent(diagnosis, risk, explanation):
    doctor_prompt = f"""
Generate a clinical report for a doctor.

Diagnosis: {diagnosis}
Risk Level: {risk}
Explanation: {explanation}
"""

    patient_prompt = f"""
Generate a patient-friendly report.

Diagnosis: {diagnosis}
Risk Level: {risk}
"""

    return {
        "doctor_report": call_llm(doctor_prompt),
        "patient_report": call_llm(patient_prompt)
    }