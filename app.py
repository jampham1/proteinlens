from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from Bio.PDB import MMCIFParser
import numpy as np
import pandas as pd
import requests
import joblib
import os

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "./data/protein_stability_model.pkl"
ENCODER_PATH = "./data/label_encoder.pkl"
PDB_DIR      = "./pdb_files"
FEATURE_COLS = ["mean_bfactor", "hydro_ratio", "bfactor_std", "charged_ratio", "avg_chain_length"]

os.makedirs(PDB_DIR, exist_ok=True)

# ── Feature extraction (same as training script) ──────────────────────────────

def fetch_and_parse(pdb_id):
    pdb_id = pdb_id.lower()
    path = f"{PDB_DIR}/{pdb_id}.cif"
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None, f"Could not download structure for {pdb_id.upper()} (RCSB status {response.status_code})"
        with open(path, "w") as f:
            f.write(response.text)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, path)
    return structure, None


def get_mean_bfactor(structure):
    b = [a.bfactor for m in structure for c in m for r in c if r.id[0]==" " for a in r]
    return np.mean(b) if b else np.nan

def get_bfactor_std(structure):
    b = [a.bfactor for m in structure for c in m for r in c if r.id[0]==" " for a in r]
    return np.std(b) if b else np.nan

def get_hydrophobic_ratio(structure):
    hydro = {"LEU","VAL","ILE","PHE","MET","TRP","PRO","ALA"}
    res = [r.resname for m in structure for c in m for r in c if r.id[0]==" "]
    return sum(1 for r in res if r in hydro)/len(res) if res else np.nan

def get_charged_ratio(structure):
    charged = {"ARG","LYS","ASP","GLU","HIS"}
    res = [r.resname for m in structure for c in m for r in c if r.id[0]==" "]
    return sum(1 for r in res if r in charged)/len(res) if res else np.nan

def get_avg_chain_length(structure):
    lengths = [sum(1 for r in c if r.id[0]==" ") for m in structure for c in m]
    return np.mean(lengths) if lengths else np.nan

def extract_features(structure):
    return {
        "mean_bfactor":     get_mean_bfactor(structure),
        "bfactor_std":      get_bfactor_std(structure),
        "hydro_ratio":      get_hydrophobic_ratio(structure),
        "charged_ratio":    get_charged_ratio(structure),
        "avg_chain_length": get_avg_chain_length(structure),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    pdb_id = data.get("pdb_id", "").strip().upper()

    if not pdb_id or len(pdb_id) != 4:
        return jsonify({"error": "Please enter a valid 4-character PDB ID."}), 400

    # Load model
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model file not found. Run the training script first."}), 500

    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    # Fetch & parse structure
    structure, err = fetch_and_parse(pdb_id)
    if err:
        return jsonify({"error": err}), 404

    # Extract features
    features = extract_features(structure)
    if any(np.isnan(v) for v in features.values()):
        return jsonify({"error": f"Could not extract valid features from {pdb_id}."}), 422

    X = pd.DataFrame([features])[FEATURE_COLS].values
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
    app.run(debug=True, port=5000)
