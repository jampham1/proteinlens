import os

DATA_DIR     = "./data"
PDB_DIR      = "./pdb_files"
FEATURES_CSV = os.path.join(DATA_DIR, "protein_features.csv")
LABELED_CSV  = os.path.join(DATA_DIR, "protein_features_labeled.csv")
MODEL_PATH   = os.path.join(DATA_DIR, "protein_stability_model.pkl")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")
FEATURE_COLS = ["mean_bfactor", "hydro_ratio", "bfactor_std",
                "charged_ratio", "avg_chain_length"]