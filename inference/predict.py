import torch


# Disease labels (must match training)
disease_labels = ["N", "D", "G", "C", "A", "H", "M", "O"]

def _extract_positive_class_probability(prob_entry):
    """Normalize LightGBM probability output to a scalar positive-class probability."""
    # 1D example: [p_negative, p_positive]
    if hasattr(prob_entry, "ndim") and prob_entry.ndim == 1:
        if len(prob_entry) > 1:
            return float(prob_entry[1])
        return float(prob_entry[0])

    # 2D example: [[p_negative, p_positive]]
    if hasattr(prob_entry, "shape") and len(prob_entry.shape) == 2 and prob_entry.shape[1] > 1:
        return float(prob_entry[0, 1])

    return float(prob_entry[0, 0])


def predict_diseases(
    model,
    lgb_model,
    img,
    threshold=0.2,
    return_confidence=False,
    return_probabilities=False,
):
    """
    Predict eye diseases and optionally return confidence scores + per-class probabilities.
    """

    # Convert image to tensor
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()

    # Feature extraction
    with torch.no_grad():
        features = model(img_tensor).cpu().numpy()

    # LightGBM probabilities
    probs = lgb_model.predict_proba(features)

    detected = []
    confidence_scores = {}
    all_probabilities = {}

    for i, prob_entry in enumerate(probs):
        label = disease_labels[i]
        prob = _extract_positive_class_probability(prob_entry)

        all_probabilities[label] = round(prob, 4)
        # Skip Normal during detection
        if label == "N":
            continue

        confidence_scores[label] = round(prob, 4)

        if prob >= threshold:
            detected.append(label)

    # If no disease detected → Normal
    if len(detected) == 0:
        detected = ["N"]
        confidence_scores = {"N": all_probabilities.get("N", 1.0)}
    result = [detected]
    if return_confidence:
        result.append(confidence_scores)
    if return_probabilities:
        result.append(all_probabilities)

    if len(result) == 1:
        return result[0]
    return tuple(result)
