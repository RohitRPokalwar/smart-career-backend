from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import pickle

from resume_parser import extract_text_from_pdf, extract_skills

app = Flask(__name__)
CORS(app)

# ===== Load AI Model and Vectorizer =====
with open("model/career_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ===== Load Skill & Learning Data =====
with open("data/role_skills.json") as f:
    role_skills = json.load(f)

with open("data/role_learning_links.json") as f:
    learning_links = json.load(f)

# ===== Health Check Route =====
@app.route("/")
def home():
    return "✅ Smart Career Advisor API is running!"

# ===== Save Report to Output Folder =====
def save_report(role, matched, missing, links):
    os.makedirs("output", exist_ok=True)
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "skills_have": matched,
        "skills_missing": missing,
        "resources": links
    }
    with open("output/last_session_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("📦 Report saved to output/last_session_report.json")

# ===== Upload Resume & Skill Match =====
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if 'file' not in request.files or 'role' not in request.form:
        return jsonify({"error": "Missing file or role"}), 400

    file = request.files['file']
    input_role = request.form['role'].strip().lower()

    # Normalize roles to avoid case mismatch
    normalized_roles = {r.lower(): r for r in role_skills}

    if input_role not in normalized_roles:
        print("❌ Invalid role:", input_role)
        print("🧠 Available roles:", list(role_skills.keys()))
        return jsonify({"error": "Invalid role"}), 400

    role_key = normalized_roles[input_role]

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save uploaded file temporarily
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)

    # Extract text and skills
    text = extract_text_from_pdf(save_path)
    known_skills = role_skills[role_key]
    resume_skills = extract_skills(text, known_skills)

    matched = resume_skills
    missing = [s for s in known_skills if s.lower() not in matched]
    resources = learning_links.get(role_key, [])

    # Save report
    save_report(role_key, matched, missing, resources)

    # Clean up uploaded file
    try:
        os.remove(save_path)
        print(f"🗑️ Deleted uploaded file: {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to delete uploaded file: {e}")

    return jsonify({
        "role": role_key,
        "skills_matched": matched,
        "skills_missing": missing,
        "resources": resources
    })

# ===== Predict Career Domain from Skills =====
@app.route("/predict", methods=["POST"])
def predict_domain():
    data = request.get_json()
    skills_text = data.get("skills", "")

    if not skills_text.strip():
        return jsonify({"error": "No skills provided"}), 400

    try:
        vectorized_input = vectorizer.transform([skills_text])
        predicted = model.predict(vectorized_input)[0]
        return jsonify({"predicted_domain": predicted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== Entry Point =====
if __name__ == "__main__":
    app.run(debug=True)
