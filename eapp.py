from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
from resume_parser import extract_text_from_pdf, extract_skills
import pickle

app = Flask(__name__)
CORS(app)
from sklearn.feature_extraction.text import CountVectorizer

# Load AI model and vectorizer
with open("model/career_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return "✅ Smart Career Advisor API is running!"

# Load skill data
with open("../data/role_skills.json") as f:
    role_skills = json.load(f)

with open("../data/role_learning_links.json") as f:
    learning_links = json.load(f)

# Utility: Save last report
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

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if 'file' not in request.files or 'role' not in request.form:
        return jsonify({"error": "Missing file or role"}), 400

    file = request.files['file']
    role = request.form['role']

    if role not in role_skills:
        return jsonify({"error": "Invalid role"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save file temporarily
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)

    # Extract resume data
    text = extract_text_from_pdf(save_path)
    known_skills = role_skills[role]
    resume_skills = extract_skills(text, known_skills)

    matched = resume_skills
    missing = [s for s in known_skills if s.lower() not in matched]
    resources = learning_links.get(role, [])

    # Save report
    save_report(role, matched, missing, resources)

    # Auto-delete file
    try:
        os.remove(save_path)
        print(f"🗑️ Deleted uploaded file: {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to delete uploaded file: {e}")

    return jsonify({
        "role": role,
        "skills_matched": matched,
        "skills_missing": missing,
        "resources": resources
    })

if __name__ == "__main__":
    app.run(debug=True)
