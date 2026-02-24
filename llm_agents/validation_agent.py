def validation_agent(diagnosis_list):
    validated = []

    for d in diagnosis_list:
        status = "Confirmed" if d["confidence"] >= 50 else "Low confidence"

        validated.append({
            "disease": d["disease"],
            "confidence": d["confidence"],
            "validation_status": status
        })

    return validated