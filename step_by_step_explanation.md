# Step-by-Step Explanation for Beginners

This guide explains the project as if you are new to data mining.

## Step 1 - Understand the Goal

We want to predict a person's activity using smartphone data.

The computer receives 561 numbers for each observation. These numbers describe movement patterns from the phone sensors.

The computer must choose one of six labels:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

This is called a **classification problem** because the answer is a category.

Because there are more than two categories, it is called **multiclass classification**.

## Step 2 - Understand the Dataset Files

The important dataset files are:

| File | Meaning |
| --- | --- |
| `X_train.txt` | Training features |
| `y_train.txt` | Training labels |
| `X_test.txt` | Test features |
| `y_test.txt` | Test labels |
| `features.txt` | Names of the 561 features |
| `activity_labels.txt` | Names of the 6 activities |

`X` means inputs.

`y` means the answer we want to predict.

## Step 3 - Load the Data

The script loads the training and test data separately.

This gives:

- `X_train`: 7352 rows and 561 columns.
- `X_test`: 2947 rows and 561 columns.
- `y_train`: the correct activity for each training row.
- `y_test`: the correct activity for each test row.

Each row is one example.

Each column is one feature.

## Step 4 - Check the Classes

The dataset has six classes.

Before training any model, we check how many examples exist in each class.

This is important because if one class had too many examples, the model might become biased toward that class.

In this dataset, the class distribution is reasonably balanced.

## Step 5 - Standardize the Features

Standardization makes features comparable.

Without standardization, a feature with large numbers could dominate a feature with smaller numbers.

The project uses `StandardScaler`.

The correct procedure is:

1. Fit the scaler on `X_train`.
2. Transform `X_train`.
3. Transform `X_test` using the same scaler.

We do not fit the scaler on `X_test`.

## Step 6 - Apply PCA

PCA reduces the number of features.

The original data has 561 features.

PCA creates new features called components.

The first components contain the most information.

In this project:

- 63 components preserve 90% of the variance.
- 102 components preserve 95% of the variance.

The project tests several PCA sizes:

- 2 components
- 20 components
- 50 components
- 100 components
- 150 components

Two components are mainly for visualization.

More components are better for classification.

## Step 7 - Train the Models

Training means the model learns from examples.

The project trains:

- Logistic Regression
- SVM RBF
- Random Forest

Each model is trained twice:

1. On original standardized features.
2. On PCA-reduced features.

This allows a fair comparison between using PCA and not using PCA.

## Step 8 - Make Predictions

After training, the model predicts the activities in the test set.

The model has not learned from the test set.

This makes the evaluation realistic.

## Step 9 - Evaluate the Models

The project uses:

- Accuracy
- Macro F1-score
- Training time
- Confusion matrix

Accuracy tells how many predictions are correct overall.

Macro F1 tells whether the model performs well across all classes.

Training time tells how expensive the model is to train.

The confusion matrix shows the exact types of mistakes.

## Step 10 - Interpret the Results

The best model was Logistic Regression on original features.

This means the original 561 engineered features are already very useful.

The model reached:

- 95.45% accuracy.
- 95.44% macro F1-score.

PCA kept good performance with 100 or 150 components, but it did not beat the original features.

The biggest confusion was between sitting and standing.

This makes sense because both activities involve little movement.

## Step 11 - Explain the Project in One Minute

This project uses smartphone sensor features to classify human activity. The dataset contains 561 preprocessed features and six activities. I kept the official train/test split to avoid data leakage. I standardized the features, applied PCA for dimensionality reduction and visualization, and trained Logistic Regression, SVM, and Random Forest models. The best model was Logistic Regression on the original standardized features, reaching 95.45% accuracy and 95.44% macro F1-score. PCA was useful for visualization and reducing dimensions, with 63 components preserving 90% variance and 102 components preserving 95% variance. The most common errors were between sitting and standing, which is logical because both activities have similar low-motion sensor patterns.

