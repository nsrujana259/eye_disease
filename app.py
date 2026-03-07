print("ACTIVE FILE LOADED:", __file__)

from flask import Flask, render_template, request, jsonify, session,Response
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
from llm_agents.llm_client import call_llm

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
        returning_from_agent = session.pop("returning_from_agent", False)
        
        if not returning_from_agent:
            session.pop("dashboard_state", None)
            session.pop("agent_outputs", None)
            session.modified = True
    
        dashboard_state = session.get("dashboard_state", {})
        return render_template(
        "index.html",
        prediction=dashboard_state.get("prediction"),
        image_url=dashboard_state.get("image_url"),
        uploaded_name=dashboard_state.get("uploaded_name"),
        has_result=bool(dashboard_state.get("prediction"))
    )


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
    explain = explanation_agent(diag)
    reports = report_agent(valid[0]["disease"], risk, explain)
    reports = {
        "patient_report_text": reports.get("patient_report_text") or reports.get("patient_report", ""),
        "doctor_report_text": reports.get("doctor_report_text") or reports.get("doctor_report", ""),
        "patient_report": reports.get("patient_report_text") or reports.get("patient_report", ""),
        "doctor_report": reports.get("doctor_report_text") or reports.get("doctor_report", ""),
    }

    # ---------- STORE IN SESSION (SAFE) ----------
    image_url = f"/{img_path}?v={int(os.path.getmtime(img_path))}"
    session["agent_outputs"] = to_builtin({
        "diagnosis": diag,
        "validation": valid,
        "risk": risk,
        "explanation": explain,
        "report": reports
    })
    uploaded_name = os.path.basename(file.filename) if file.filename else "captured.jpg"
    session["dashboard_state"] = {
        "prediction": predictions,
        "image_url": image_url,
        "uploaded_name": uploaded_name
    }
    session.modified = True

    return jsonify(prediction=predictions)


@app.route("/agent/<agent_name>")
def agent_page(agent_name):

    agent_outputs = session.get("agent_outputs")

    if not agent_outputs:
        return "Please upload an image first."

    if agent_name not in agent_outputs:
        return "Invalid agent."
    
    session["returning_from_agent"] = True

    return render_template(
        "agent.html",
        agent_name=agent_name.title(),
        agent_output=agent_outputs[agent_name]
    )
@app.route("/download/patient_report")
def download_patient_report():
    agent_outputs = session.get("agent_outputs", {})
    report_data = agent_outputs.get("report", {})
    text = report_data.get("patient_report_text") or report_data.get("patient_report")

    if not text:
        return "Please upload an image first.", 400

    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=patient_report.txt"}
    )


@app.route("/download/doctor_report")
def download_doctor_report():
    agent_outputs = session.get("agent_outputs", {})
    report_data = agent_outputs.get("report", {})
    text = report_data.get("doctor_report_text") or report_data.get("doctor_report")

    if not text:
        return "Please upload an image first.", 400

    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=doctor_report.txt"}
    )
@app.route("/translate_report", methods=["POST"])
def translate_report():
    payload = request.get_json(silent=True) or {}
    report_type = payload.get("report_type")
    target_language = payload.get("language", "English")

    if report_type not in {"patient", "doctor"}:
        return jsonify(error="Invalid report type"), 400

    agent_outputs = session.get("agent_outputs", {})
    report_data = agent_outputs.get("report", {})

    source_text = (
        report_data.get("patient_report_text") or report_data.get("patient_report", "")
        if report_type == "patient"
        else report_data.get("doctor_report_text") or report_data.get("doctor_report", "")
    )

    if not source_text:
        return jsonify(error="Report not available. Please upload an image first."), 400

    if target_language == "English":
        return jsonify(translated_text=source_text)

    translations = session.get("report_translations", {})
    report_cache = translations.setdefault(report_type, {})
    if target_language in report_cache:
        return jsonify(translated_text=report_cache[target_language])

    prompt = f"""
Translate the following medical screening report into {target_language}.
Keep medical terms accurate and unchanged where needed.
Keep the meaning and structure consistent.
Return only the translated report text.

Report:
{source_text}
"""

    translated_text = call_llm(prompt)
    report_cache[target_language] = translated_text
    session["report_translations"] = translations
    session.modified = True

    return jsonify(translated_text=translated_text)
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
    