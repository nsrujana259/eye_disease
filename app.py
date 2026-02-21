print("ACTIVE FILE LOADED:", __file__)

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import torch
import joblib
import cv2

# ===============================
# ML IMPORTS
# ===============================
from preprocessing.basic_preprocess import basic_preprocess
from unet.unet_segment import unet_segment
from models.densenet_loader import load_densenet
from inference.predict import predict_diseases

# ===============================
# AGENT IMPORTS
# ===============================
from llm_agents.diagnosis_agent import diagnosis_agent
from llm_agents.validation_agent import validation_agent
from llm_agents.risk_agent import risk_assessment_agent
from llm_agents.explanation_agent import explanation_agent
from llm_agents.report_agent import report_agent


# ===============================
# APP INITIALIZATION
# ===============================
app = Flask(__name__)

app.config.update(
    SECRET_KEY="eye-ai-secret-key",
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=os.path.abspath("./flask_session"),
    SESSION_PERMANENT=False,
    SESSION_USE_SIGNER=True,
    SESSION_COOKIE_NAME="eye_ai_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

Session(app)

os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
os.makedirs("static", exist_ok=True)


# ===============================
# DEVICE + MODEL LOADING
# ===============================
device = torch.device("cpu")

densenet = load_densenet(device)
lgb_model = joblib.load("models/lightgbm_classifier (1).pkl")


# ===============================
# SAFE SERIALIZER
# ===============================
def to_builtin(value):
    """
    Convert numpy / torch / complex objects into JSON-safe types
    """
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify(error="No image received"), 400

    # Reset session cleanly
    session.clear()

    file = request.files["image"]
    img_path = os.path.join("static", "captured.jpg")
    file.save(img_path)

    # ---------- PREPROCESS ----------
    img = basic_preprocess(img_path)
    img = unet_segment(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # ---------- MODEL PREDICTION ----------
    predictions, confidence_scores = predict_diseases(
        densenet, lgb_model, img, return_confidence=True
    )

    ml_output = {
        "diseases": predictions,
        "confidence": confidence_scores
    }

    # ---------- AGENT PIPELINE ----------
    diag = diagnosis_agent(ml_output)
    valid = validation_agent(diag)
    risk = risk_assessment_agent(valid)
    explain = explanation_agent(valid)
    reports = report_agent(valid[0]["disease"], risk, explain)

    # ---------- STORE IN SESSION (SAFE) ----------
    session["agent_outputs"] = to_builtin({
        "diagnosis": diag,
        "validation": valid,
        "risk": risk,
        "explanation": explain,
        "report": reports
    })

    session.modified = True

    return jsonify(prediction=predictions)


@app.route("/agent/<agent_name>")
def agent_page(agent_name):

    agent_outputs = session.get("agent_outputs")

    if not agent_outputs:
        return "Please upload an image first."

    if agent_name not in agent_outputs:
        return "Invalid agent."

    return render_template(
        "agent.html",
        agent_name=agent_name.title(),
        agent_output=agent_outputs[agent_name]
    )


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    print("Starting Flask server...")
    print("Session dir:", app.config["SESSION_FILE_DIR"])
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )