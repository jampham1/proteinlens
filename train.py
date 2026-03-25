from Bio.PDB import MMCIFParser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import requests
import shutil
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR     = "./data"
FEATURES_CSV = os.path.join(DATA_DIR, "protein_features.csv")
LABELED_CSV  = os.path.join(DATA_DIR, "protein_features_labeled.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Move existing CSVs into data folder if still in root
for filename in ["protein_features.csv", "protein_features_labeled.csv"]:
    old_path = f"./{filename}"
    new_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(old_path) and not os.path.exists(new_path):
        shutil.move(old_path, new_path)
        print(f"Moved {filename} → {DATA_DIR}/")

# ── PDB IDs ───────────────────────────────────────────────────────────────────

pdb_ids = [
    "1C5E", "4BOQ", "8EI0", "9QFS", "2ESQ",
    "9GTE", "6STW", "7HLC", "7HLD", "7HLK",
    "7HLN", "7HLR", "7HLU", "7HLV", "7HLW",
    "7HM0", "7HM1", "7HM2", "7HM4", "7HM5",
    "8JVD", "8JVE", "8JUC", "3NIL", "1TD0",
    "6S98", "6JWJ", "7TB1", "9DYB", "3R42",
    "6US8", "3UPV", "4BOU", "1XC1", "2CLW",
    "5D1L", "5D1M", "2OJ6", "3RWA", "7HLG",
    "3NIM", "1QYN", "6R75", "3NII", "6S96",
    "2G37", "6STU", "1VQ8", "1VQL", "3CC2",
    "9R1P", "3LCA", "1N7Z", "2QUX", "5LYN",
    "3BJQ", "3CNC", "3AIH", "8J9Q", "6QPA",
    "6ZC5", "5XAN", "1FOU", "1JNB", "3AQP",
    "4BWF", "2ZJQ", "1NKW", "3CF5", "6X0R",
    "6OJ2", "1XMQ", "1XMO", "1HNX", "1J5E",
    "2HHH", "1K73", "1IBK", "2ZM6", "7AZS",
    "1K8A", "5DM7", "2ZJR", "7TJM", "6YFH",
    "7JOQ", "1XBP", "1UF2", "1NWX", "2FT1",
    "6O3M", "1N34", "4IOC", "2F4V", "1W2B",
    "4V4J", "5WNS", "2YXQ", "2YXR", "4DUZ",
    "4JI6", "4V8O", "4DV5", "3L2O", "6QH3",
    "4H4L", "1FX3", "8C7E", "5IB7", "1YJW",
    "1VQ4", "1VQ5", "7NPI", "1YJ9", "1QVG",
    "1Q81", "1Q82", "2QEX", "3G71", "3CC7"
]

# ── Feature extraction functions ──────────────────────────────────────────────

def fetch_and_parse(pdb_id):
    """Obtains protein structure from RCSB database"""
    pdb_id = pdb_id.lower()
    os.makedirs("./pdb_files", exist_ok=True)
    path = f"./pdb_files/{pdb_id}.cif"
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[WARN] Could not download {pdb_id} (status {response.status_code})")
            return None
        with open(path, "w") as f:
            f.write(response.text)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, path)
    return structure


def get_mean_bfactor(structure):
    """Mean B-factor — measures average atomic displacement (flexibility)."""
    b_factors = [
        atom.bfactor
        for model in structure
        for chain in model
        for residue in chain
        if residue.id[0] == " "
        for atom in residue
    ]
    return np.mean(b_factors) if b_factors else np.nan


def get_bfactor_std(structure):
    """Std deviation of B-factors — high variance = flexible/unstable regions."""
    b_factors = [
        atom.bfactor
        for model in structure
        for chain in model
        for residue in chain
        if residue.id[0] == " "
        for atom in residue
    ]
    return np.std(b_factors) if b_factors else np.nan


def get_hydrophobic_ratio(structure):
    """Fraction of hydrophobic residues — core packing drives stability."""
    hydrophobic = {"LEU", "VAL", "ILE", "PHE", "MET", "TRP", "PRO", "ALA"}
    residues = [
        residue.resname
        for model in structure
        for chain in model
        for residue in chain
        if residue.id[0] == " "
    ]
    if not residues:
        return np.nan
    return sum(1 for r in residues if r in hydrophobic) / len(residues)


def get_charged_residue_ratio(structure):
    """Fraction of charged residues — salt bridges contribute to stability."""
    charged = {"ARG", "LYS", "ASP", "GLU", "HIS"}
    residues = [
        residue.resname
        for model in structure
        for chain in model
        for residue in chain
        if residue.id[0] == " "
    ]
    if not residues:
        return np.nan
    return sum(1 for r in residues if r in charged) / len(residues)


def get_avg_chain_length(structure):
    """Average chain length — larger proteins tend to have more stable cores."""
    chain_lengths = [
        sum(1 for res in chain if res.id[0] == " ")
        for model in structure
        for chain in model
    ]
    return np.mean(chain_lengths) if chain_lengths else np.nan


def extract_features(pdb_id):
    """Extracts specific features of protein structure from target pdb IDs"""
    path = f"./pdb_files/{pdb_id.lower()}.cif"
    if not os.path.exists(path):
        print(f"[WARN] Missing file for {pdb_id}")
        return None
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id.lower(), path)
    return {
        "pdb_id":           pdb_id,
        "mean_bfactor":     get_mean_bfactor(structure),
        "bfactor_std":      get_bfactor_std(structure),
        "hydro_ratio":      get_hydrophobic_ratio(structure),
        "charged_ratio":    get_charged_residue_ratio(structure),
        "avg_chain_length": get_avg_chain_length(structure),
    }


# ── Step 1: Extract features (skip if CSV already exists) ─────────────────────

if os.path.exists(FEATURES_CSV):
    print("Features already extracted, loading from CSV...")
    df = pd.read_csv(filepath_or_buffer=FEATURES_CSV)
else:
    print("Extracting features from PDB files...")
    cif_ids = [f.replace(".cif", "") for f in os.listdir("./pdb_files") if f.endswith(".cif")]
    records = []
    for pid in cif_ids:
        print(f"Processing {pid}...")
        record = extract_features(pid)
        if record:
            records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(df)} proteins to {FEATURES_CSV}")

print(df.head())
print(df.describe())


# ── Labeling function ─────────────────────────────────────────────────────────

def get_quality_label(pdb_id):
    """Assigns good/medium/bad label from RCSB crystallographic validation metrics."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"[WARN] Could not fetch validation data for {pdb_id}")
        return None, None

    data = response.json()

    try:
        geom = data["pdbx_vrpt_summary_geometry"]
        if isinstance(geom, list):
            geom = geom[0]

        diff = data.get("pdbx_vrpt_summary_diffraction", {})
        if isinstance(diff, list):
            diff = diff[0]

        clashscore       = geom.get("clashscore")
        rama_outliers    = geom.get("percent_ramachandran_outliers")
        rotamer_outliers = geom.get("percent_rotamer_outliers")
        rsrz_outliers    = diff.get("percent_rsrzoutliers")

        scores = []
        if clashscore is not None:
            scores.append(2 if clashscore < 10 else (1 if clashscore < 25 else 0))
        if rama_outliers is not None:
            scores.append(2 if rama_outliers < 0.5 else (1 if rama_outliers < 2.0 else 0))
        if rotamer_outliers is not None:
            scores.append(2 if rotamer_outliers < 1.0 else (1 if rotamer_outliers < 5.0 else 0))
        if rsrz_outliers is not None:
            scores.append(2 if rsrz_outliers < 5.0 else (1 if rsrz_outliers < 10 else 0))

        if len(scores) < 2:
            print(f"[WARN] Too few valid metrics for {pdb_id}, skipping")
            return None, None

        composite = np.mean(scores)
        label = "good" if composite >= 1.5 else ("medium" if composite >= 0.75 else "bad")
        return composite, label

    except KeyError as e:
        print(f"[WARN] Missing validation field for {pdb_id}: {e}")
        return None, None


# ── Step 2: Attach labels (skip if labeled CSV already exists) ─────────────────

if os.path.exists(LABELED_CSV):
    print("Labels already fetched, loading from CSV...")
    df = pd.read_csv(filepath_or_buffer=LABELED_CSV)
else:
    print("Fetching validation labels from RCSB...")
    composite_scores, quality_labels = [], []
    for pid in df["pdb_id"]:
        print(f"Fetching validation data for {pid}...")
        score, label = get_quality_label(pid)
        composite_scores.append(score)
        quality_labels.append(label)
    df["composite_score"] = composite_scores
    df["quality_label"]   = quality_labels
    df = df.dropna(subset=["quality_label"])
    df.to_csv(LABELED_CSV, index=False)
    print(f"Saved {len(df)} labeled proteins to {LABELED_CSV}")

print(f"\nLabel distribution:\n{df['quality_label'].value_counts()}")
print(f"Total labeled proteins: {len(df)}")


# ── Step 3: Prepare data for training ─────────────────────────────────────────

df = pd.read_csv(filepath_or_buffer=LABELED_CSV).dropna()

FEATURE_COLS = ["mean_bfactor", "hydro_ratio", "bfactor_std", "charged_ratio", "avg_chain_length"]
LABEL_COL    = "quality_label"

X = df[FEATURE_COLS].values

le = LabelEncoder()
y  = le.fit_transform(df[LABEL_COL])
print(f"\nClasses: {le.classes_}")
print(f"Label distribution:\n{df[LABEL_COL].value_counts()}\n")


# ── Step 4: Cross-validate models ─────────────────────────────────────────────

models = {
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== Cross-validation results ===")
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model)
    ])
    results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "f1_weighted"],
        return_train_score=True
    )
    print(f"\n{name}:")
    print(f"  Train accuracy : {results['train_accuracy'].mean():.3f} ± {results['train_accuracy'].std():.3f}")
    print(f"  Test accuracy  : {results['test_accuracy'].mean():.3f} ± {results['test_accuracy'].std():.3f}")
    print(f"  Test F1        : {results['test_f1_weighted'].mean():.3f} ± {results['test_f1_weighted'].std():.3f}")


# ── Step 4b: Tune Random Forest to reduce overfitting ─────────────────────────

param_grid = {
    "model__n_estimators":     [50, 100, 200],
    "model__max_depth":        [2, 3, 4, None],
    "model__min_samples_leaf": [2, 4, 6, 8],
    "model__max_features":     ["sqrt", "log2"],
}

pipeline_tune = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(random_state=42))
])

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

grid_search = GridSearchCV(
    pipeline_tune, param_grid,
    cv=cv_inner, scoring="f1_weighted", n_jobs=-1, verbose=0
)

results_tuned = cross_validate(
    grid_search, X, y,
    cv=cv_outer,
    scoring=["accuracy", "f1_weighted"],
    return_train_score=True,
    return_estimator=True
)

print("\n=== Tuned Random Forest (nested CV) ===")
print(f"  Train accuracy : {results_tuned['train_accuracy'].mean():.3f} ± {results_tuned['train_accuracy'].std():.3f}")
print(f"  Test accuracy  : {results_tuned['test_accuracy'].mean():.3f} ± {results_tuned['test_accuracy'].std():.3f}")
print(f"  Test F1        : {results_tuned['test_f1_weighted'].mean():.3f} ± {results_tuned['test_f1_weighted'].std():.3f}")

best_params = results_tuned["estimator"][-1].best_params_
print(f"\n  Best parameters found: {best_params}")


# ── Step 5: Fit final model using best parameters ─────────────────────────────

grid_search.fit(X, y)
final_pipeline = grid_search.best_estimator_

y_pred = final_pipeline.predict(X)

print("\n=== Classification report (tuned model, full dataset) ===")
print(classification_report(y, y_pred, target_names=le.classes_))

importances = final_pipeline.named_steps["model"].feature_importances_
print("=== Feature importances (tuned model) ===")
for name, imp in zip(FEATURE_COLS, importances):
    print(f"  {name}: {imp:.3f}")


# ── Step 5b: Improve bad class recall with SMOTE oversampling ─────────────────

sm = SMOTE(random_state=42, k_neighbors=4)  # k=4 since bad class only has 25 samples
X_resampled, y_resampled = sm.fit_resample(X, y)

print("\n=== Class distribution after SMOTE ===")
resampled_counts = pd.Series(le.inverse_transform(y_resampled)).value_counts()
print(resampled_counts)

# Retrain tuned model on resampled data
grid_search_smote = GridSearchCV(
    pipeline_tune, param_grid,
    cv=cv_inner, scoring="f1_weighted", n_jobs=-1, verbose=0
)

results_smote = cross_validate(
    grid_search_smote, X_resampled, y_resampled,
    cv=cv_outer,
    scoring=["accuracy", "f1_weighted"],
    return_train_score=True,
    return_estimator=True
)

print("\n=== Tuned Random Forest + SMOTE (nested CV) ===")
print(f"  Train accuracy : {results_smote['train_accuracy'].mean():.3f} ± {results_smote['train_accuracy'].std():.3f}")
print(f"  Test accuracy  : {results_smote['test_accuracy'].mean():.3f} ± {results_smote['test_accuracy'].std():.3f}")
print(f"  Test F1        : {results_smote['test_f1_weighted'].mean():.3f} ± {results_smote['test_f1_weighted'].std():.3f}")

# Fit final SMOTE model and evaluate
grid_search_smote.fit(X_resampled, y_resampled)
final_pipeline_smote = grid_search_smote.best_estimator_

# Evaluate on ORIGINAL data (not resampled) for honest metrics
y_pred_smote = final_pipeline_smote.predict(X)

print("\n=== Classification report (SMOTE model, original data) ===")
print(classification_report(y, y_pred_smote, target_names=le.classes_))

importances_smote = final_pipeline_smote.named_steps["model"].feature_importances_
print("=== Feature importances (SMOTE model) ===")
for name, imp in zip(FEATURE_COLS, importances_smote):
    print(f"  {name}: {imp:.3f}")


# ── Step 6: Plots ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].barh(FEATURE_COLS, importances, color=["#1D9E75", "#3B8BD4", "#E24B4A", "#EF9F27", "#534AB7"])
axes[0].set_xlabel("Importance")
axes[0].set_title("Feature importances")

ConfusionMatrixDisplay.from_predictions(
    y, y_pred, display_labels=le.classes_,
    cmap="Blues", ax=axes[1]
)
axes[1].set_title("Confusion matrix")

colors = {"bad": "#E24B4A", "medium": "#EF9F27", "good": "#1D9E75"}
for label in df[LABEL_COL].unique():
    mask = df[LABEL_COL] == label
    axes[2].scatter(
        df.loc[mask, "mean_bfactor"],
        df.loc[mask, "hydro_ratio"],
        label=label, color=colors.get(label, "gray"), alpha=0.7
    )
axes[2].set_xlabel("Mean B-factor")
axes[2].set_ylabel("Hydrophobic ratio")
axes[2].set_title("Feature space by label")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "model_results.png"), dpi=150)
plt.show()

print("\nDone — results saved to model_results.png")


# ── Step 6: Save final SMOTE model ────────────────────────────────────────────

MODEL_PATH = "./data/protein_stability_model.pkl"
joblib.dump(final_pipeline_smote, MODEL_PATH)
joblib.dump(le, "./data/label_encoder.pkl")
print(f"Model saved to {MODEL_PATH}")

# ── Step 7: Predict stability for a new protein ───────────────────────────────

def predict_stability(pdb_id):
    """Predict crystallographic quality label for a new protein by PDB ID."""
    structure = fetch_and_parse(pdb_id)
    if structure is None:
        return None

    record = extract_features(pdb_id)
    if record is None:
        return None

    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load("./data/label_encoder.pkl")

    features = pd.DataFrame([record])[FEATURE_COLS].values  # add .values here
    pred     = model.predict(features)
    proba    = model.predict_proba(features)

    label      = encoder.inverse_transform(pred)[0]
    confidence = max(proba[0])

    print(f"\nPDB ID    : {pdb_id.upper()}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1%}")
    print("Class probabilities:")
    for cls, prob in zip(encoder.classes_, proba[0]):
        print(f"  {cls:8s}: {prob:.1%}")

    return label

# Test on a protein not in your training set
predict_stability("1TIM")