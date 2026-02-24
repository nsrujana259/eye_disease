from llm_agents.llm_client import call_llm

def report_agent(diagnosis, risk, explanation):
    doctor_prompt = f"""
Generate a clinical report for a doctor.
Use structured sections: Diagnosis, Confidence, Risk, Remarks.
Include a short disclaimer that this is an AI-based screening result and not a final diagnosis.
Diagnosis: {diagnosis}
Risk Level: {risk}
Explanation: {explanation}
"""

    patient_prompt = f"""
Generate a patient-friendly report in simple language.
Avoid medical jargon.
Include a short disclaimer that this is an AI-based screening result and doctor consultation is required.

Diagnosis: {diagnosis}
Risk Level: {risk}
"""
  
    doctor_report_text = call_llm(doctor_prompt)
    patient_report_text = call_llm(patient_prompt)
    return {
        "doctor_report_text": doctor_report_text,
        "patient_report_text": patient_report_text,
        # Backward-compatible keys
        "doctor_report": doctor_report_text,
        "patient_report": patient_report_text,
    }