# Project 17 - Human Activity Recognition Using Smartphones

## 1. Project Idea in Simple Words

This project is about teaching a computer to recognize what a person is doing using data collected from a smartphone.

The phone has sensors such as:

- Accelerometer: measures movement and acceleration.
- Gyroscope: measures rotation and orientation changes.

When someone walks, sits, stands, lies down, or goes upstairs, the phone records different movement patterns. The goal of this project is to use those patterns to automatically predict the activity.

The six activities are:

| Class | Meaning |
| --- | --- |
| WALKING | The person is walking normally |
| WALKING_UPSTAIRS | The person is walking upstairs |
| WALKING_DOWNSTAIRS | The person is walking downstairs |
| SITTING | The person is sitting |
| STANDING | The person is standing |
| LAYING | The person is lying down |

So the main question is:

> Can we automatically recognize a human activity from smartphone sensor signals?

The answer from our results is yes. The best model reached about **95.45% accuracy** and **95.44% macro F1-score**.

---

## 2. What Data Mining Means Here

Data mining means finding useful patterns in data.

In this project, the raw information comes from smartphone sensors. Instead of looking at the sensor signals manually, we use machine learning models to learn patterns automatically.

The general data mining workflow is:

1. Understand the problem.
2. Load the dataset.
3. Understand the variables and target classes.
4. Prepare the data.
5. Train machine learning models.
6. Evaluate the models.
7. Interpret the results.

This project follows that same workflow.

---

## 3. Dataset Description

The dataset used is:

**Human Activity Recognition Using Smartphones**

Source:

**UCI Machine Learning Repository**

The dataset contains observations collected from people carrying smartphones while performing activities. Each observation has:

- 561 numerical features.
- 1 activity label.

The 561 features are not raw sensor readings directly. They are already preprocessed features extracted from accelerometer and gyroscope signals.

Examples of feature ideas include:

- Mean values.
- Standard deviation values.
- Frequency-domain measurements.
- Body acceleration features.
- Gravity acceleration features.
- Gyroscope features.

We do not need to manually create these 561 features because they are already provided in the dataset.

---

## 4. Train and Test Split

The dataset already gives two separate parts:

| Set | Number of observations | Number of features |
| --- | ---: | ---: |
| Training set | 7352 | 561 |
| Test set | 2947 | 561 |

The training set is used to teach the model.

The test set is used only at the end to check whether the model works on new data.

This separation is very important.

If we mix training and test data, the model may indirectly see the answers before evaluation. This is called **data leakage**. Data leakage gives results that look good but are not trustworthy.

In this project, the original train/test split was respected.

---

## 5. Target Variable

The target variable is the activity we want to predict.

There are six possible classes:

| Activity | Train count | Test count | Total |
| --- | ---: | ---: | ---: |
| LAYING | 1407 | 537 | 1944 |
| SITTING | 1286 | 491 | 1777 |
| STANDING | 1374 | 532 | 1906 |
| WALKING | 1226 | 496 | 1722 |
| WALKING_DOWNSTAIRS | 986 | 420 | 1406 |
| WALKING_UPSTAIRS | 1073 | 471 | 1544 |

The classes are reasonably balanced. This means no activity completely dominates the dataset.

This is good because the model has enough examples from each class.

Figure:

`outputs/figures/class_distribution.png`

---

## 6. Why Standardization Is Needed

The dataset has 561 numerical features. Some features may have different scales.

For example, one feature might vary between -1 and 1, while another might vary between -100 and 100.

Some machine learning methods are sensitive to scale, especially:

- Logistic Regression.
- SVM.
- PCA.

Standardization transforms each feature so that:

- The mean becomes 0.
- The standard deviation becomes 1.

In simple words, standardization puts all features on a comparable scale.

Important rule:

The scaler is fitted only on the training data. Then the same scaler is applied to the test data.

This avoids data leakage.

---

## 7. What PCA Is

PCA means **Principal Component Analysis**.

It is a dimensionality reduction method.

Dimensionality means the number of features. Here, the original data has 561 features, so it is high-dimensional.

PCA creates new variables called **principal components**.

These components are combinations of the original features. PCA orders them from most informative to least informative, based on how much variance they explain.

In simple words:

> PCA tries to summarize a large number of features using fewer new features while losing as little information as possible.

---

## 8. Why PCA Is Useful in This Project

PCA is useful for two reasons.

First, it helps visualization. Humans cannot visualize 561 dimensions, but we can visualize 2 dimensions. So we use the first two PCA components to create a 2D plot.

Second, it can reduce the number of features before classification. Instead of training models on 561 original features, we can train them on fewer PCA components, such as 20, 50, 100, or 150.

This can make training faster and sometimes reduce noise.

However, PCA may also remove useful information if too few components are kept.

---

## 9. PCA Results

The PCA results were:

| PCA information | Value |
| --- | ---: |
| Variance explained by PC1 | 50.78% |
| Variance explained by PC2 | 6.58% |
| Components needed for 90% variance | 63 |
| Components needed for 95% variance | 102 |

This means the first component alone explains a very large part of the information in the dataset.

But the first two components together are not enough to preserve everything. They are useful for visualization, but not enough for maximum classification performance.

Figures:

- `outputs/figures/pca_variance.png`
- `outputs/figures/pca_2d_projection.png`

Important rule:

PCA was fitted only on `X_train`. Then it was used to transform `X_test`.

PCA was not fitted on test data.

---

## 10. Models Used

Three classification models were used.

### 10.1 Logistic Regression

Despite the name, Logistic Regression is used for classification.

It tries to find boundaries that separate the classes.

For this project, it is a strong baseline because the dataset is already well prepared and contains engineered features.

### 10.2 Support Vector Machine

SVM tries to separate classes using a decision boundary.

The RBF kernel allows the model to learn nonlinear boundaries.

This is useful when the classes are not separated by simple straight lines.

### 10.3 Random Forest

Random Forest is a model made of many decision trees.

Each tree makes a prediction, and the forest combines their votes.

Random Forest is often powerful, but it can take more training time.

---

## 11. Evaluation Metrics

The project uses three main evaluation values.

### 11.1 Accuracy

Accuracy means:

> Out of all predictions, how many were correct?

Formula:

```text
accuracy = correct predictions / total predictions
```

Accuracy is easy to understand, but it can be misleading if classes are very imbalanced.

### 11.2 Macro F1-Score

Macro F1-score calculates the F1-score for each class and then averages them equally.

It is important because it gives each activity the same importance.

In this project, macro F1 is required and useful because we care about all six activities, not just the most frequent ones.

### 11.3 Training Time

Training time shows how long the model takes to learn.

This matters because a model with slightly lower accuracy but much faster training can sometimes be more practical.

---

## 12. Model Results

The best results were:

| Setting | Model | Accuracy | Macro F1 | Training time |
| --- | --- | ---: | ---: | ---: |
| Original features | Logistic Regression | 0.9545 | 0.9544 | 1.4301 sec |
| Original features | SVM RBF | 0.9542 | 0.9533 | 3.1695 sec |
| PCA 150 components | SVM RBF | 0.9471 | 0.9459 | 0.7344 sec |
| PCA 150 components | Logistic Regression | 0.9444 | 0.9437 | 1.3819 sec |
| PCA 100 components | SVM RBF | 0.9396 | 0.9384 | 0.5220 sec |
| PCA 100 components | Logistic Regression | 0.9338 | 0.9323 | 1.4798 sec |
| Original features | Random Forest | 0.9304 | 0.9283 | 7.9486 sec |

The best model overall was:

**Logistic Regression with the original 561 standardized features**

It achieved:

- Accuracy: **95.45%**
- Macro F1-score: **95.44%**
- Training time: **1.43 seconds**

This is a strong result.

---

## 13. Original Features vs PCA

The original features performed best.

This makes sense because the original 561 features were already carefully engineered from the sensor signals.

PCA reduced the number of dimensions and still kept strong performance, especially with 100 or 150 components.

For example:

- SVM with 150 PCA components reached 94.71% accuracy.
- SVM with 100 PCA components reached 93.96% accuracy.

This shows that PCA can reduce dimensionality while keeping much of the predictive information.

However, using only 2 PCA components produced much weaker results:

| Setting | Model | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| PCA 2 components | SVM RBF | 0.5728 | 0.5544 |
| PCA 2 components | Logistic Regression | 0.5663 | 0.5191 |
| PCA 2 components | Random Forest | 0.5226 | 0.5187 |

This proves that the first two PCA components are good for visualization, but not enough for final classification.

---

## 14. Confusion Matrix Interpretation

A confusion matrix shows where the model is correct and where it makes mistakes.

Rows represent the true activity.

Columns represent the predicted activity.

The diagonal values are correct predictions.

Values outside the diagonal are errors.

For the best model, the biggest confusions were:

| True activity | Predicted as | Number of mistakes |
| --- | --- | ---: |
| SITTING | STANDING | 58 |
| WALKING_UPSTAIRS | WALKING | 24 |
| WALKING_DOWNSTAIRS | WALKING_UPSTAIRS | 18 |
| STANDING | SITTING | 14 |
| WALKING_DOWNSTAIRS | WALKING | 6 |
| LAYING | STANDING | 5 |

The most important confusion is between **SITTING** and **STANDING**.

This is logical because both activities have low movement. A smartphone may record very similar sensor patterns when a person is sitting still or standing still.

The model also sometimes confuses walking upstairs, walking downstairs, and normal walking. This also makes sense because all three are dynamic walking activities.

Figure:

`outputs/figures/confusion_original_logistic_regression.png`

---

## 15. What the 2D PCA Plot Means

The PCA 2D plot places every training observation on a graph using only the first two principal components.

Each point is one observation.

The color represents the activity.

If activities form separate groups, that means PCA found useful structure.

In this project, the plot helps show that:

- Some activities are easier to separate.
- Static activities and dynamic activities show different patterns.
- Some overlap remains, especially between similar activities.

The plot is useful for understanding the data visually, but it is not enough by itself to build the best classifier.

---

## 16. Why the Best Model Is Logistic Regression

At first, it may seem surprising that Logistic Regression performs best.

But this result is reasonable.

The dataset does not contain raw messy signals. It contains 561 engineered features that already summarize useful signal information.

When features are well designed, a simpler model can perform extremely well.

Logistic Regression was:

- Very accurate.
- Fast to train.
- Stable.
- Easy to interpret compared with more complex models.

That makes it a strong final choice.

---

## 17. Main Conclusion

This project shows that human activities can be recognized accurately using smartphone sensor features.

The best result was obtained using Logistic Regression on the original standardized features, with about **95.45% accuracy** and **95.44% macro F1-score**.

PCA was useful for visualization and dimensionality reduction. It showed that 63 components are enough to preserve 90% of the variance, and 102 components are enough to preserve 95% of the variance.

However, PCA did not improve the best classification result. The original 561 features gave the strongest performance.

The main errors happened between similar activities:

- Sitting and standing.
- Walking, walking upstairs, and walking downstairs.

These mistakes are understandable because similar physical movements produce similar sensor patterns.

---

## 18. What Was Done Correctly

This project respects the required methodology:

- The official UCI HAR dataset was used.
- `X_train`, `y_train`, `X_test`, `y_test`, `features`, and `activity_labels` were loaded.
- The six activity classes were studied.
- The 561-feature dimension was verified.
- Class distribution was visualized.
- Features were standardized.
- PCA was fitted only on training data.
- PCA explained variance was calculated.
- A 2D PCA projection was generated.
- Logistic Regression, SVM, and Random Forest were trained.
- Models were compared before and after PCA.
- Accuracy, macro F1, training time, and confusion matrices were reported.
- The most confused activities were interpreted.

---

## 19. Files Generated

Important files:

| File or folder | Purpose |
| --- | --- |
| `src/run_har_project.py` | Main Python script |
| `outputs/report.md` | Short automatic report |
| `final_report_beginner_friendly.md` | Detailed final report |
| `step_by_step_explanation.md` | Learning guide explaining each step |
| `oral_presentation_notes.md` | Simple presentation script |
| `outputs/tables/results_summary.csv` | Model comparison table |
| `outputs/tables/pca_explained_variance.csv` | PCA explained variance table |
| `outputs/figures/` | All required figures |

---

## 20. Final Answer to the Problem

Yes, it is possible to automatically recognize human activity using smartphone sensor data.

The best model recognized the six activities with high performance, reaching around **95% accuracy**.

The project also shows that dimensionality reduction with PCA can reduce the number of features, but the original engineered features gave the best final classification performance.

