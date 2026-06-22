from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = RAW_DIR / "UCI HAR Dataset"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


@dataclass
class Dataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    features: list[str]
    activity_labels: dict[int, str]


def ensure_dirs() -> None:
    for path in [RAW_DIR, FIG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_and_extract_dataset() -> None:
    zip_path = RAW_DIR / "uci_har_dataset.zip"
    inner_zip_path = RAW_DIR / "UCI HAR Dataset.zip"
    if (DATASET_DIR / "features.txt").exists():
        return

    if not zip_path.exists():
        print("Downloading UCI HAR Dataset...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(RAW_DIR)

    if inner_zip_path.exists() and not (DATASET_DIR / "features.txt").exists():
        with zipfile.ZipFile(inner_zip_path, "r") as archive:
            archive.extractall(RAW_DIR)


def make_unique_feature_names(raw_features: pd.Series) -> list[str]:
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for feature in raw_features.astype(str):
        count = counts.get(feature, 0)
        unique_names.append(feature if count == 0 else f"{feature}_{count + 1}")
        counts[feature] = count + 1
    return unique_names


def load_dataset() -> Dataset:
    features_df = pd.read_csv(
        DATASET_DIR / "features.txt",
        sep=r"\s+",
        header=None,
        names=["index", "feature"],
        engine="python",
    )
    features = make_unique_feature_names(features_df["feature"])

    labels_df = pd.read_csv(
        DATASET_DIR / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["id", "activity"],
        engine="python",
    )
    activity_labels = dict(zip(labels_df["id"], labels_df["activity"]))

    x_train = pd.read_csv(
        DATASET_DIR / "train" / "X_train.txt",
        sep=r"\s+",
        header=None,
        names=features,
        engine="python",
    )
    x_test = pd.read_csv(
        DATASET_DIR / "test" / "X_test.txt",
        sep=r"\s+",
        header=None,
        names=features,
        engine="python",
    )
    y_train_ids = pd.read_csv(DATASET_DIR / "train" / "y_train.txt", header=None)[0]
    y_test_ids = pd.read_csv(DATASET_DIR / "test" / "y_test.txt", header=None)[0]

    y_train = y_train_ids.map(activity_labels)
    y_test = y_test_ids.map(activity_labels)

    return Dataset(x_train, y_train, x_test, y_test, features, activity_labels)


def save_class_distribution(data: Dataset) -> pd.DataFrame:
    distribution = (
        pd.concat(
            [
                data.y_train.value_counts().rename("train"),
                data.y_test.value_counts().rename("test"),
            ],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    distribution["total"] = distribution["train"] + distribution["test"]
    distribution.to_csv(TABLE_DIR / "class_distribution.csv")

    ax = distribution[["train", "test"]].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Class Distribution in Provided Train/Test Split")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Number of observations")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=180)
    plt.close()
    return distribution


def fit_scaler_and_pca(data: Dataset) -> tuple[StandardScaler, PCA, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data.x_train)
    x_test_scaled = scaler.transform(data.x_test)

    pca = PCA(random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)

    variance_table = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": cumulative,
        }
    )
    variance_table.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(variance_table["component"], variance_table["cumulative_variance"])
    plt.axhline(0.90, color="tab:green", linestyle="--", label=f"90% at {components_90} components")
    plt.axhline(0.95, color="tab:red", linestyle="--", label=f"95% at {components_95} components")
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_variance.png", dpi=180)
    plt.close()

    projection = pd.DataFrame(
        {
            "PC1": x_train_pca[:, 0],
            "PC2": x_train_pca[:, 1],
            "activity": data.y_train,
        }
    )
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=projection,
        x="PC1",
        y="PC2",
        hue="activity",
        s=18,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("Training Data Projected on the First Two PCA Components")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_2d_projection.png", dpi=180)
    plt.close()

    metadata = {
        "components_for_90_percent_variance": components_90,
        "components_for_95_percent_variance": components_95,
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
    }
    (TABLE_DIR / "pca_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return scaler, pca, x_train_pca, x_test_pca


def models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs"),
        "SVM RBF": SVC(kernel="rbf", C=10, gamma="scale"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted activity")
    plt.ylabel("True activity")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display_df = df.copy()
    display_df = display_df.reset_index() if display_df.index.name or not isinstance(display_df.index, pd.RangeIndex) else display_df
    headers = [str(column) for column in display_df.columns]
    rows = []
    for _, row in display_df.iterrows():
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def evaluate_model(
    name: str,
    model: object,
    x_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series,
    x_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
    setting: str,
) -> dict[str, object]:
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    report.to_csv(TABLE_DIR / f"classification_report_{setting}_{safe_name}.csv")
    plot_confusion(
        y_test,
        y_pred,
        labels,
        f"{name} Confusion Matrix ({setting})",
        FIG_DIR / f"confusion_{setting}_{safe_name}.png",
    )

    return {
        "setting": setting,
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_seconds": train_time,
    }


def run_experiments(data: Dataset, x_train_pca: np.ndarray, x_test_pca: np.ndarray) -> pd.DataFrame:
    labels = list(data.activity_labels.values())
    rows: list[dict[str, object]] = []

    for name, estimator in models().items():
        original_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        rows.append(
            evaluate_model(
                name,
                original_pipeline,
                data.x_train,
                data.y_train,
                data.x_test,
                data.y_test,
                labels,
                "original",
            )
        )

    pca_component_options = [2, 20, 50, 100, 150]
    for n_components in pca_component_options:
        for name, estimator in models().items():
            rows.append(
                evaluate_model(
                    name,
                    estimator,
                    x_train_pca[:, :n_components],
                    data.y_train,
                    x_test_pca[:, :n_components],
                    data.y_test,
                    labels,
                    f"pca_{n_components}",
                )
            )

    results = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    results.to_csv(TABLE_DIR / "results_summary.csv", index=False)
    return results


def write_report(data: Dataset, distribution: pd.DataFrame, results: pd.DataFrame) -> None:
    pca_metadata = json.loads((TABLE_DIR / "pca_metadata.json").read_text(encoding="utf-8"))
    best = results.iloc[0]
    best_original = results[results["setting"] == "original"].iloc[0]
    best_pca = results[results["setting"].str.startswith("pca_")].iloc[0]
    distribution_table = dataframe_to_markdown(distribution.rename_axis("activity"))
    top_results_table = dataframe_to_markdown(results.head(10))

    report = f"""# Human Activity Recognition Using Smartphones

## 1. Problem

The goal is to classify a person's activity using smartphone sensor features. Each observation has 561 numerical features extracted from accelerometer and gyroscope signals. This is a multiclass classification problem with six classes.

## 2. Dataset

- Source: UCI Machine Learning Repository, Human Activity Recognition Using Smartphones.
- Training observations: {len(data.x_train)}
- Test observations: {len(data.x_test)}
- Number of features: {data.x_train.shape[1]}
- Number of classes: {data.y_train.nunique()}

The provided train/test split was kept exactly as given. This matters because mixing the two sets would create data leakage and make the evaluation unreliable.

## 3. Class Distribution

The classes are reasonably balanced, although not perfectly identical in size.

{distribution_table}

See `outputs/figures/class_distribution.png`.

## 4. Standardization

The features are standardized before Logistic Regression, SVM, and PCA. Standardization transforms each feature to have mean 0 and standard deviation 1 based only on the training set. The test set is transformed using those training statistics.

## 5. PCA Analysis

PCA is used to reduce the 561-dimensional feature space into fewer components while preserving as much variance as possible.

- PC1 explained variance: {pca_metadata["pc1_variance"]:.3f}
- PC2 explained variance: {pca_metadata["pc2_variance"]:.3f}
- Components needed for 90% variance: {pca_metadata["components_for_90_percent_variance"]}
- Components needed for 95% variance: {pca_metadata["components_for_95_percent_variance"]}

Important methodological rule: PCA was fitted on `X_train` only, then applied to `X_test`. PCA was not adjusted on the test set.

See:

- `outputs/figures/pca_variance.png`
- `outputs/figures/pca_2d_projection.png`

## 6. Model Comparison

Three models were compared:

- Logistic Regression
- SVM with RBF kernel
- Random Forest

Each model was trained on the original standardized features and then on PCA-reduced features with several component counts.

Top results:

{top_results_table}

Best overall model:

- Setting: {best["setting"]}
- Model: {best["model"]}
- Accuracy: {best["accuracy"]:.4f}
- Macro F1: {best["macro_f1"]:.4f}
- Training time: {best["train_time_seconds"]:.3f} seconds

Best original-feature result:

- Model: {best_original["model"]}
- Accuracy: {best_original["accuracy"]:.4f}
- Macro F1: {best_original["macro_f1"]:.4f}

Best PCA result:

- Setting: {best_pca["setting"]}
- Model: {best_pca["model"]}
- Accuracy: {best_pca["accuracy"]:.4f}
- Macro F1: {best_pca["macro_f1"]:.4f}

## 7. Interpretation

The first two PCA components give a useful visual summary, but two dimensions are not enough to perfectly separate all activities. Dynamic activities such as WALKING, WALKING_UPSTAIRS, and WALKING_DOWNSTAIRS tend to separate better from static activities. The most difficult confusions usually happen between SITTING and STANDING because both are low-motion postures and can produce similar sensor summaries.

Using more PCA components usually improves performance compared with only two components. However, the original 561 features may still perform very strongly because the dataset already contains carefully engineered signal features.

## 8. Conclusion

Smartphone sensor features can classify human activity with high performance. PCA is useful for visualization and dimensionality reduction, but the best number of components should be chosen by comparing macro F1, accuracy, and training time. Macro F1 is especially important because it evaluates performance across all classes rather than only rewarding the majority classes.

## 9. Required Figures and Tables

- Class distribution: `outputs/figures/class_distribution.png`
- PCA explained variance: `outputs/figures/pca_variance.png`
- PCA 2D projection: `outputs/figures/pca_2d_projection.png`
- Confusion matrices: `outputs/figures/confusion_*.png`
- Results table: `outputs/tables/results_summary.csv`
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    download_and_extract_dataset()
    data = load_dataset()

    print(f"Loaded train shape: {data.x_train.shape}")
    print(f"Loaded test shape: {data.x_test.shape}")

    distribution = save_class_distribution(data)
    _, _, x_train_pca, x_test_pca = fit_scaler_and_pca(data)
    results = run_experiments(data, x_train_pca, x_test_pca)
    write_report(data, distribution, results)

    print("\nDone. Main outputs:")
    print(f"- {OUTPUT_DIR / 'report.md'}")
    print(f"- {TABLE_DIR / 'results_summary.csv'}")
    print(f"- {FIG_DIR}")


if __name__ == "__main__":
    main()
