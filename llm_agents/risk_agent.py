def risk_assessment_agent(validated):
    primary = validated[0]["disease"]

    if primary == "Normal Retina":
        return "Low Risk"
    else:
        return "High Risk"