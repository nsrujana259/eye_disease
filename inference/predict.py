import torch
import numpy as np

# Disease labels (must match training)
disease_labels = ["N", "D", "G", "C", "A", "H", "M", "O"]

def predict_diseases(model, lgb_model, img, threshold=0.2, return_confidence=False):
    """
    Predict eye diseases and optionally return confidence scores.
    """

    # Convert image to tensor
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    # Feature extraction
    with torch.no_grad():
        features = model(img).cpu().numpy()

    # LightGBM probabilities
    probs = lgb_model.predict_proba(features)

    detected = []
    confidence_scores = {}

    for i, p in enumerate(probs):
        label = disease_labels[i]

        # Skip Normal during detection
        if label == "N":
            continue

        # Handle probability shape safely
        if p.ndim == 1:
            prob = float(p[0])
        else:
            prob = float(p[:, 1][0])

        confidence_scores[label] = round(prob, 2)

        if prob >= threshold:
            detected.append(label)

    # If no disease detected → Normal
    if len(detected) == 0:
        detected = ["N"]
        confidence_scores = {"N": 1.0}

    if return_confidence:
        return detected, confidence_scores
    else:
        return detected
