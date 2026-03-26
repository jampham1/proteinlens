from proteinlens.features import extract_features
from proteinlens.labels import get_quality_label
from config import (
    DATA_DIR, FEATURES_CSV, LABELED_CSV,
     MODEL_PATH, ENCODER_PATH
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

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


    # ── Step 7: Save final SMOTE model ────────────────────────────────────────────

    joblib.dump(final_pipeline_smote, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()