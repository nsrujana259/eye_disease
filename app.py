print("ACTIVE FILE LOADED:", __file__)

from flask import Flask, render_template, request, jsonify, session, Response, redirect
from flask_session import Session
import os
import torch
import joblib
import cv2
import bcrypt

# ===============================
# ML IMPORTS
# ===============================
from preprocessing.basic_preprocess import basic_preprocess
from unet.unet_segment import unet_segment
from models.densenet_loader import load_densenet
from inference.predict import predict_diseases
from inference.gradcam import generate_gradcam_overlay

# ===============================
# AGENT IMPORTS
# ===============================
from llm_agents.diagnosis_agent import diagnosis_agent
from llm_agents.validation_agent import validation_agent
from llm_agents.risk_agent import risk_assessment_agent
from llm_agents.explanation_agent import explanation_agent
from llm_agents.report_agent import report_agent
from llm_agents.llm_client import call_llm


import pymysql

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",   # your MySQL password
        database="eye_ai_users",
        cursorclass=pymysql.cursors.DictCursor
    )
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
)

Session(app)

os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
os.makedirs("static", exist_ok=True)

# ===============================
# MYSQL CONFIG
# ===============================
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "root"
app.config["MYSQL_DB"] = "eye_ai_users"


# ===============================
# DEVICE + MODEL LOADING
# ===============================
device = torch.device("cpu")

densenet = load_densenet(device)
lgb_model = joblib.load("models/lightgbm_classifier (1).pkl")

DISEASE_LABELS = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension-related Retinopathy",
    "M": "Myopia",
    "O": "Other Retinal Abnormalities",
}

# ===============================
# SAFE SERIALIZER
# ===============================
def to_builtin(value):

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
# AUTHENTICATION ROUTES
# ===============================

@app.route("/login_page")
def login_page():
    return render_template("login.html")


@app.route("/register_page")
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        existing = cur.fetchone()

        if existing:
            cur.close()
            conn.close()
            return jsonify({"message": "Email already registered"}), 400

        cur.execute(
            "INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
            (username, email, hashed_password)
        )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "User registered successfully"})

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"message": "Server error"}), 500

@app.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    email = data["email"]
    password = data["password"]

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cur.fetchone()

    cur.close()
    conn.close()

    if user:

        if bcrypt.checkpw(
            password.encode("utf-8"),
            user["password"].encode("utf-8")
        ):
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return jsonify({"message": "Login successful"})

        else:
            return jsonify({"message": "Invalid password"})

    return jsonify({"message": "User not found"})

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login_page")


# ===============================
# DASHBOARD
# ===============================
@app.route("/")
def index():

    if "user_id" not in session:
        return redirect("/login_page")

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


# ===============================
# IMAGE UPLOAD + AI ANALYSIS
# ===============================
@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify(error="No image received"), 400

    file = request.files["image"]

    img_path = os.path.join("static", "captured.jpg")
    file.save(img_path)

    img = basic_preprocess(img_path)
    img = unet_segment(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    predictions, confidence_scores, all_probabilities = predict_diseases(
        densenet,
        lgb_model,
        img,
        return_confidence=True,
        return_probabilities=True,
    )

    ml_output = {
        "diseases": predictions,
        "confidence": confidence_scores
    }

    diag = diagnosis_agent(ml_output)
    valid = validation_agent(diag)
    risk = risk_assessment_agent(valid)
    explain = explanation_agent(diag)
    reports = report_agent(valid[0]["disease"], risk, explain)

    heatmap_path = os.path.join("static", "gradcam_heatmap.jpg")

    focus_label = predictions[0] if predictions else "N"

    focus_index = list(DISEASE_LABELS.keys()).index(focus_label)

    generate_gradcam_overlay(densenet, img, heatmap_path, focus_index=focus_index)

    disease_probabilities = {
        DISEASE_LABELS.get(code, code): round(score * 100, 2)
        for code, score in all_probabilities.items()
    }

    image_url = f"/{img_path}?v={int(os.path.getmtime(img_path))}"
    heatmap_url = f"/{heatmap_path}?v={int(os.path.getmtime(heatmap_path))}"

    session["agent_outputs"] = to_builtin({
        "diagnosis": {
            "results": diag,
            "probabilities": disease_probabilities,
            "heatmap_url": heatmap_url,
            "image_url": image_url,
        },
        "validation": valid,
        "risk": risk,
        "explanation": explain,
        "report": reports
    })

    uploaded_name = os.path.basename(file.filename)

    session["dashboard_state"] = {
        "prediction": predictions,
        "image_url": image_url,
        "uploaded_name": uploaded_name
    }

    session.modified = True

    return jsonify(prediction=predictions)


# ===============================
# AGENT PAGES
# ===============================
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


# ===============================
# REPORT DOWNLOAD
# ===============================
@app.route("/download/patient_report")
def download_patient_report():

    agent_outputs = session.get("agent_outputs", {})
    report_data = agent_outputs.get("report", {})

    text = report_data.get("patient_report_text") or report_data.get("patient_report")

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

    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=doctor_report.txt"}
    )


# ===============================
# REPORT TRANSLATION
# ===============================
@app.route("/translate_report", methods=["POST"])
def translate_report():

    payload = request.get_json()

    report_type = payload.get("report_type")
    target_language = payload.get("language", "English")

    agent_outputs = session.get("agent_outputs", {})
    report_data = agent_outputs.get("report", {})

    source_text = (
        report_data.get("patient_report_text")
        if report_type == "patient"
        else report_data.get("doctor_report_text")
    )

    if target_language == "English":
        return jsonify(translated_text=source_text)

    prompt = f"""
Translate the following medical screening report into {target_language}.
Keep medical terms accurate.

Report:
{source_text}
"""

    translated_text = call_llm(prompt)

    return jsonify(translated_text=translated_text)


# ===============================
# RUN SERVER
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