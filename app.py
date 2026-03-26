from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from proteinlens.features import fetch_and_parse, extract_features
from config import MODEL_PATH, ENCODER_PATH, FEATURE_COLS, PDB_DIR
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder="static")
CORS(app)

os.makedirs(PDB_DIR, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data   = request.get_json()
    pdb_id = data.get("pdb_id", "").strip().upper()

    if not pdb_id or len(pdb_id) != 4:
        return jsonify({"error": "Please enter a valid 4-character PDB ID."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model file not found. Run train.py first."}), 500

    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    features = extract_features(pdb_id)
    if features is None:
        return jsonify({"error": f"Could not fetch or parse structure for {pdb_id}."}), 404

    features.pop("pdb_id", None)  # ← now correctly placed after the None check

    if any(np.isnan(v) for v in features.values()):
        return jsonify({"error": f"Could not extract valid features from {pdb_id}."}), 422

    X     = pd.DataFrame([features])[FEATURE_COLS].values
    pred  = model.predict(X)
    proba = model.predict_proba(X)[0]
    label = encoder.inverse_transform(pred)[0]

    return jsonify({
        "pdb_id":     pdb_id,
        "label":      label,
        "confidence": float(max(proba)),
        "probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(encoder.classes_, proba)
        },
        "features": {k: round(float(v), 4) for k, v in features.items()},
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)