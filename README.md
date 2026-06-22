# Project 17 - Human Activity Recognition Using Smartphones

This project answers the question:

> Can we automatically recognize a human activity from preprocessed smartphone sensor signals?

The dataset is the UCI **Human Activity Recognition Using Smartphones** dataset. It contains 561 engineered features extracted from accelerometer and gyroscope signals, and the goal is to classify each observation into one of six activities:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## What This Project Produces

Running the main script creates:

- `outputs/figures/class_distribution.png`
- `outputs/figures/pca_variance.png`
- `outputs/figures/pca_2d_projection.png`
- `outputs/figures/confusion_original_*.png`
- `outputs/figures/confusion_pca_*.png`
- `outputs/tables/results_summary.csv`
- `outputs/tables/classification_report_*.csv`
- `outputs/report.md`

The report covers all required project points: loading the provided train/test split, checking 561 features, standardization, PCA, classification before/after PCA, accuracy, macro F1, training time, confusion matrices, and interpretation of the most confused activities.

## How To Run

From this folder:

```powershell
& "C:\Users\halas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r requirements.txt
& "C:\Users\halas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" src\run_har_project.py
```

If you are using a normal local Python installation, the commands are:

```powershell
python -m pip install -r requirements.txt
python src\run_har_project.py
```

## Important Method Rule

PCA is fitted only on the training data. The test data is transformed using the already-fitted PCA object. This avoids data leakage and respects the project instruction: **do not fit PCA on the test set**.
