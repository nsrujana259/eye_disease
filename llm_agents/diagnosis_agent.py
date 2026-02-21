DISEASE_MAP = {
    "N": "Normal Retina",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension-related Retinopathy",
    "M": "Myopia",
    "O": "Other Retinal Abnormalities"
}

def diagnosis_agent(ml_output):
    diseases = ml_output["diseases"]
    confidence = ml_output["confidence"]

    structured = []
    for d in diseases:
        structured.append({
            "disease": DISEASE_MAP[d],
            "confidence": round(confidence[d] * 100, 2)
        })

    return structured