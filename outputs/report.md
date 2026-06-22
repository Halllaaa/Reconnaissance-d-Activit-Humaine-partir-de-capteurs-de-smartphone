# Human Activity Recognition Using Smartphones

## 1. Problem

The goal is to classify a person's activity using smartphone sensor features. Each observation has 561 numerical features extracted from accelerometer and gyroscope signals. This is a multiclass classification problem with six classes.

## 2. Dataset

- Source: UCI Machine Learning Repository, Human Activity Recognition Using Smartphones.
- Training observations: 7352
- Test observations: 2947
- Number of features: 561
- Number of classes: 6

The provided train/test split was kept exactly as given. This matters because mixing the two sets would create data leakage and make the evaluation unreliable.

## 3. Class Distribution

The classes are reasonably balanced, although not perfectly identical in size.

| activity | train | test | total |
| --- | --- | --- | --- |
| LAYING | 1407 | 537 | 1944 |
| SITTING | 1286 | 491 | 1777 |
| STANDING | 1374 | 532 | 1906 |
| WALKING | 1226 | 496 | 1722 |
| WALKING_DOWNSTAIRS | 986 | 420 | 1406 |
| WALKING_UPSTAIRS | 1073 | 471 | 1544 |

See `outputs/figures/class_distribution.png`.

## 4. Standardization

The features are standardized before Logistic Regression, SVM, and PCA. Standardization transforms each feature to have mean 0 and standard deviation 1 based only on the training set. The test set is transformed using those training statistics.

## 5. PCA Analysis

PCA is used to reduce the 561-dimensional feature space into fewer components while preserving as much variance as possible.

- PC1 explained variance: 0.508
- PC2 explained variance: 0.066
- Components needed for 90% variance: 63
- Components needed for 95% variance: 102

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

| index | setting | model | accuracy | macro_f1 | train_time_seconds |
| --- | --- | --- | --- | --- | --- |
| 0 | original | Logistic Regression | 0.9545 | 0.9544 | 1.4301 |
| 1 | original | SVM RBF | 0.9542 | 0.9533 | 3.1695 |
| 16 | pca_150 | SVM RBF | 0.9471 | 0.9459 | 0.7344 |
| 15 | pca_150 | Logistic Regression | 0.9444 | 0.9437 | 1.3819 |
| 13 | pca_100 | SVM RBF | 0.9396 | 0.9384 | 0.5220 |
| 12 | pca_100 | Logistic Regression | 0.9338 | 0.9323 | 1.4798 |
| 2 | original | Random Forest | 0.9304 | 0.9283 | 7.9486 |
| 10 | pca_50 | SVM RBF | 0.9169 | 0.9157 | 0.3912 |
| 9 | pca_50 | Logistic Regression | 0.9152 | 0.9141 | 2.1849 |
| 14 | pca_100 | Random Forest | 0.8877 | 0.8842 | 4.3341 |

Best overall model:

- Setting: original
- Model: Logistic Regression
- Accuracy: 0.9545
- Macro F1: 0.9544
- Training time: 1.430 seconds

Best original-feature result:

- Model: Logistic Regression
- Accuracy: 0.9545
- Macro F1: 0.9544

Best PCA result:

- Setting: pca_150
- Model: SVM RBF
- Accuracy: 0.9471
- Macro F1: 0.9459

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
