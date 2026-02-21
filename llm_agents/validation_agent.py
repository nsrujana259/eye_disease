def validation_agent(diagnosis_list):
    validated = []

    for d in diagnosis_list:
        if d["confidence"] >= 50:
            validated.append(d)

    if not validated:
        validated.append({
            "disease": "Normal Retina",
            "confidence": 100
        })

    return validated