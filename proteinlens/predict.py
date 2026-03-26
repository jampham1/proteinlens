import joblib
import pandas as pd
from config import MODEL_PATH, ENCODER_PATH, FEATURE_COLS
from proteinlens.features import fetch_and_parse, extract_features


def predict_stability(pdb_id):
    """Predict crystallographic quality label for a new protein by PDB ID."""
    structure, err = fetch_and_parse(pdb_id)
    if err:
        print(f"[ERROR] {err}")
        return None

    record = extract_features(pdb_id)
    if record is None:
        return None

    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    features   = pd.DataFrame([record])[FEATURE_COLS].values
    pred       = model.predict(features)
    proba      = model.predict_proba(features)
    label      = encoder.inverse_transform(pred)[0]
    confidence = max(proba[0])

    print(f"\nPDB ID    : {pdb_id.upper()}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1%}")
    print("Class probabilities:")
    for cls, prob in zip(encoder.classes_, proba[0]):
        print(f"  {cls:8s}: {prob:.1%}")

    return label