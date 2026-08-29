import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - used only when dependency is missing
    XGBClassifier = None

sns.set_theme(style="whitegrid")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]


def load_medical_dataset():
    """Load the structured medical dataset and define the disease target."""
    import kagglehub

    path = Path(kagglehub.dataset_download("imtkaggleteam/diabetes"))
    csv_path = path / "diabetes.csv"
    print("Path to dataset files:", path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Loaded dataset:", csv_path)

    df = df.drop(columns=["bp.2s", "bp.2d", "id"], errors="ignore")
    df = df.dropna(subset=["glyhb"]).reset_index(drop=True)
    df["diabetes"] = (df["glyhb"] >= 6.5).astype(int)
    return df


def build_model_dict():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "SVM": CalibratedClassifierCV(
            estimator=SVC(
                class_weight="balanced",
                kernel="rbf",
                C=2.0,
                gamma="scale",
                random_state=42,
            ),
            method="sigmoid",
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
    }

    if XGBClassifier is None:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )

    models["XGBoost"] = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
    )
    return models


def build_preprocessor(X, feature_names=None):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if "diabetes" in numeric_cols:
        numeric_cols.remove("diabetes")
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if feature_names is not None:
        numeric_cols = [col for col in feature_names if col in numeric_cols]
        categorical_cols = [col for col in feature_names if col in categorical_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols


def plot_class_balance(df):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(
        x=df["diabetes"].map({0: "Non-diabetic", 1: "Diabetic"}),
        hue=df["diabetes"].map({0: "Non-diabetic", 1: "Diabetic"}),
        palette=["#4C72B0", "#DD8452"],
        legend=False,
    )
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )
    plt.title("Class Balance (HbA1c >= 6.5% threshold)")
    plt.xlabel("")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_class_balance.png", dpi=150)
    plt.close()


def plot_feature_distributions(df, numeric_cols):
    key_feats = ["chol", "stab.glu", "hdl", "ratio", "age", "weight", "bp.1s", "waist"]
    key_feats = [c for c in key_feats if c in numeric_cols]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_feats):
        sns.histplot(
            data=df,
            x=col,
            hue=df["diabetes"].map({0: "Non-diabetic", 1: "Diabetic"}),
            kde=True,
            ax=axes[i],
            palette=["#4C72B0", "#DD8452"],
            element="step",
        )
        axes[i].set_title(col)
        axes[i].set_ylabel("")
    plt.suptitle("Feature Distributions by Diabetes Status", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df, numeric_cols):
    corr_df = df[numeric_cols + ["diabetes"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix (numeric features + target)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_boxplots(df, numeric_cols):
    key_feats = ["chol", "stab.glu", "hdl", "ratio", "age", "weight", "bp.1s", "waist"]
    key_feats = [c for c in key_feats if c in numeric_cols]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_feats):
        sns.boxplot(
            data=df,
            x=df["diabetes"].map({0: "Non-diabetic", 1: "Diabetic"}),
            y=col,
            hue=df["diabetes"].map({0: "Non-diabetic", 1: "Diabetic"}),
            ax=axes[i],
            palette=["#4C72B0", "#DD8452"],
            legend=False,
        )
        axes[i].set_title(col)
        axes[i].set_xlabel("")
    plt.suptitle("Key Features by Diabetes Status", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor, models):
    results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"]))
        print("ROC-AUC:", results[name]["roc_auc"])

    return results, fitted_pipelines


def save_metric_comparison(results):
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in res.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
        for name, res in results.items()
    }).T
    metrics_df.to_csv(f"{OUT}/model_metrics.csv")
    print("\n", metrics_df)

    plt.figure(figsize=(8, 5))
    metrics_df.plot(kind="bar", ax=plt.gca(), colormap="tab10")
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_model_comparison.png", dpi=150)
    plt.close()

    return metrics_df


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Non-diabetic", "Diabetic"],
            yticklabels=["Non-diabetic", "Diabetic"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(6, 5.5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/07_roc_curves.png", dpi=150)
    plt.close()


def plot_random_forest_importance(fitted_pipelines, numeric_cols, categorical_cols):
    rf_pipe = fitted_pipelines["Random Forest"]
    ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names
    importances = rf_pipe.named_steps["clf"].feature_importances_
    imp_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=imp_series.values[:12],
        y=imp_series.index[:12],
        hue=imp_series.index[:12],
        palette="viridis",
        legend=False,
    )
    plt.title("Top Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT}/08_feature_importance.png", dpi=150)
    plt.close()


def main():
    df = load_medical_dataset()
    features_df = df.drop(columns=["glyhb"])

    target_counts = df["diabetes"].value_counts()
    print("Class balance:\n", target_counts, "\n")

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "diabetes"]
    categorical_cols = features_df.select_dtypes(include=["object", "string"]).columns.tolist()
    print("Numeric features:", numeric_cols)
    print("Categorical features:", categorical_cols)

    plot_class_balance(df)
    plot_feature_distributions(df, numeric_cols)
    plot_correlation_heatmap(df, numeric_cols)
    plot_boxplots(df, numeric_cols)

    X = features_df.drop(columns=["diabetes"])
    y = features_df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    models = build_model_dict()
    results, fitted_pipelines = evaluate_models(X_train, X_test, y_train, y_test, preprocessor, models)

    save_metric_comparison(results)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_random_forest_importance(fitted_pipelines, numeric_cols, categorical_cols)

    train_out = X_train.copy()
    train_out["diabetes"] = y_train
    test_out = X_test.copy()
    test_out["diabetes"] = y_test
    train_out.to_csv(f"{OUT}/train_set.csv", index=False)
    test_out.to_csv(f"{OUT}/test_set.csv", index=False)

    print("\nDone. All outputs written to", OUT)


if __name__ == "__main__":
    main()
