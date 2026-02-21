from llm_agents.diagnosis_agent import diagnosis_agent
from llm_agents.validation_agent import validation_agent
from llm_agents.risk_agent import risk_assessment_agent
from llm_agents.explanation_agent import explanation_agent
from llm_agents.report_agent import report_agent

# Example ML output
ml_output = {
    "diseases": ["D", "G"],
    "confidence": {"D": 0.78, "G": 0.64}
}

diag = diagnosis_agent(ml_output)
valid = validation_agent(diag)
risk = risk_assessment_agent(valid)
explain = explanation_agent(valid)
reports = report_agent(
    valid[0]["disease"],
    risk,
    explain
)

print(reports["doctor_report"])
print(reports["patient_report"])
