import os
import json
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, jsonify, make_response
)
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import pandas as pd

# ---------- CONFIG ----------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
FEEDBACK_FOLDER = os.path.join(APP_ROOT, "feedback")
MODEL_PATH = os.path.join(APP_ROOT, "final_model.h5")
ALLOWED_EXT = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)
ADMIN_PASSWORD = "admin123"     # change before sharing publicly

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "dermavision_secret_key_v2"

# ---------- MODEL LOAD (cached) ----------
model = None
def load_model_once():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        except TypeError:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# Attempt load on start (prints to console)
try:
    load_model_once()
    print("✅ Model loaded.")
except Exception as e:
    print("⚠️ Model load failed at startup:", e)

# ---------- HELPERS ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_pil_image(pil_img):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

CLASS_NAMES = [
    "Melanoma (MEL)",
    "Melanocytic Nevus (NV)",
    "Basal Cell Carcinoma (BCC)",
    "Benign Keratosis (BKL)"
]

def get_recent_uploads(limit=3):
    files = []
    for fname in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(path):
            files.append((path, os.path.getmtime(path)))
    files_sorted = sorted(files, key=lambda x: x[1], reverse=True)
    # return relative urls for display
    urls = [url_for("uploaded_file", filename=os.path.basename(p[0])) for p in files_sorted[:limit]]
    return urls

# ---------- ROUTES ----------
@app.route("/")
def index():
    recent = get_recent_uploads()
    return render_template("index.html", title="Home", recent=recent)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/detection", methods=["GET", "POST"])
def detection():
    if request.method == "POST":
        # form fields
        age = request.form.get("age", type=int)
        sex = request.form.get("sex", type=str)
        medical_history = request.form.get("medical_history", "")
        if "image" not in request.files:
            flash("Please upload an image.", "warning")
            return redirect(url_for("detection"))
        file = request.files["image"]
        if file.filename == "":
            flash("Please choose a file.", "warning")
            return redirect(url_for("detection"))
        if not allowed_file(file.filename):
            flash("Allowed: png, jpg, jpeg", "danger")
            return redirect(url_for("detection"))

        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # model prediction
        try:
            pil_img = Image.open(save_path)
            x = preprocess_pil_image(pil_img)
            model_local = load_model_once()
            preds = model_local.predict(x)
            pred_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds) * 100)
            predicted_class = CLASS_NAMES[pred_idx]
        except Exception as e:
            flash(f"Prediction error: {e}", "danger")
            return redirect(url_for("detection"))

        # descriptions & guidelines (detailed)
        if "Melanoma" in predicted_class:
            description = ("Melanoma is a serious skin cancer arising from pigment-producing cells (melanocytes). "
                           "Early detection and biopsy when suspicious improves outcomes significantly.")
            guidelines = [
                "Consult a dermatologist for dermoscopy & possible biopsy.",
                "Follow the ABCDE rule: Asymmetry, Border, Color, Diameter, Evolution.",
                "Use SPF 50+, avoid tanning beds and prolonged sun exposure.",
                "Self-check monthly and seek professional checks annually or per clinician."
            ]
            severity_hint = "Higher urgency — seek immediate specialist evaluation."
        elif "Nevus" in predicted_class:
            description = ("Melanocytic nevus (mole) — usually benign. Monitor for changes in color, shape, size or symptoms.")
            guidelines = [
                "Monitor with photos and report rapid changes.",
                "Protect from sun and avoid intentional tanning.",
                "If many atypical moles or family history, schedule dermatology review."
            ]
            severity_hint = "Usually low urgency; monitor closely."
        elif "Basal" in predicted_class:
            description = ("Basal cell carcinoma (BCC) — the most common skin cancer, slow-growing but locally invasive.")
            guidelines = [
                "See dermatologist for removal options (excision, curettage, topical therapy).",
                "Avoid repetitive sun damage and use protective clothing.",
                "Annual skin exams are recommended for those with sun damage."
            ]
            severity_hint = "Moderate urgency — early treatment recommended."
        else:  # BKL
            description = ("Benign keratosis-like lesion — common non-cancerous growth related to sun or age.")
            guidelines = [
                "Benign in most cases; remove if symptomatic or for cosmetic reasons.",
                "Keep skin moisturized and protected from UV.",
                "Seek review if rapid growth, bleeding or irritation occurs."
            ]
            severity_hint = "Low urgency."

        # age-specific note
        if age is None:
            age_note = ""
        elif age < 20:
            age_note = ("Under 20: lower baseline risk. Emphasize sun safety. Seek review for symptomatic or rapidly changing lesions.")
        elif age <= 60:
            age_note = ("Age 20–60: maintain regular self-checks; consider annual professional exam if risk factors exist.")
        else:
            age_note = ("Age 60+: increased risk. Recommend annual full-body skin exam and prompt evaluation for any suspicious lesion.")

        # prepare result dict (for JSON download)
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "age": age,
            "sex": sex,
            "medical_history": medical_history,
            "description": description,
            "guidelines": guidelines,
            "age_note": age_note,
            "severity_hint": severity_hint
        }

        # render results
        recent = get_recent_uploads()
        return render_template(
            "detection.html",
            title="Detection",
            uploaded_image=url_for("uploaded_file", filename=filename),
            result=result,
            recent=recent
        )
    # GET
    recent = get_recent_uploads()
    return render_template("detection.html", title="Detection", recent=recent)

@app.route("/download_result/<filename>")
def download_result(filename):
    # find feedback/result csv? Instead we search uploads and return JSON stub if exists
    # We'll create a JSON by reading file prefix (timestamp) — for simplicity, regenerate nothing
    # This route expects a query param with result JSON data (or could be generated in session)
    # But we'll support a simple JSON download by reading a stored JSON if present.
    json_path = os.path.join(FEEDBACK_FOLDER, f"{filename}.json")
    if os.path.exists(json_path):
        return send_from_directory(FEEDBACK_FOLDER, f"{filename}.json", as_attachment=True)
    else:
        # Not found — return 404 JSON
        return jsonify({"error": "Result JSON not found"}), 404

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        if not (name and email and message):
            flash("Please fill all fields.", "warning")
            return redirect(url_for("contact"))
        csv_path = os.path.join(FEEDBACK_FOLDER, "feedbacks.csv")
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "email": email,
            "message": message
        }
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        else:
            df = pd.DataFrame([entry])
        df.to_csv(csv_path, index=False)
        flash("Thank you — feedback recorded.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html", title="Contact")

@app.route("/insights")
def insights():
    return render_template("insights.html", title="Insights")

@app.route("/about")
def about():
    return render_template("about_developer.html", title="Developer")

# Simple admin to view feedback CSV (password protected)
@app.route("/admin_feedback", methods=["GET", "POST"])
def admin_feedback():
    pwd = request.args.get("pwd") or request.form.get("pwd")
    if request.method == "POST":
        pwd = request.form.get("pwd", "")
    if pwd != ADMIN_PASSWORD:
        return render_template("admin_login.html", title="Admin - Login", error = (pwd is not None and pwd != ""))
    csv_path = os.path.join(FEEDBACK_FOLDER, "feedbacks.csv")
    feedbacks = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feedbacks = df.to_dict(orient="records")
    return render_template("admin_view.html", title="Admin - Feedback", feedbacks=feedbacks)

# ---------- RUN ----------
if __name__ == "__main__":
    # force model load (already attempted above)
    try:
        load_model_once()
    except Exception as e:
        print("Model load error:", e)
    app.run(host="0.0.0.0", port=5000, debug=True)
