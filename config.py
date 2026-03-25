# config.py
DATA_DIR      = "../data"
PDB_DIR       = "../pdb_files"
FEATURES_CSV  = "./data/protein_features.csv"
LABELED_CSV   = "./data/protein_features_labeled.csv"
MODEL_PATH    = "../data/protein_stability_model.pkl"
ENCODER_PATH  = "../data/label_encoder.pkl"
FEATURE_COLS  = ["mean_bfactor", "hydro_ratio", "bfactor_std",
                 "charged_ratio", "avg_chain_length"]