# Oral Presentation Notes

## Opening

My project is about human activity recognition using smartphone sensor data.

The objective is to predict what a person is doing from preprocessed accelerometer and gyroscope features.

The six possible activities are walking, walking upstairs, walking downstairs, sitting, standing, and laying.

## Dataset

I used the UCI Human Activity Recognition Using Smartphones dataset.

The dataset contains 561 features and six classes.

There are 7352 training observations and 2947 test observations.

I kept the original train/test split because mixing them would create data leakage.

## Method

First, I loaded the training data, test data, feature names, and activity labels.

Then I checked the distribution of the six activities.

After that, I standardized the features because PCA, Logistic Regression, and SVM are sensitive to feature scale.

Then I applied PCA.

PCA was fitted only on the training set, and the test set was transformed afterward.

This is important because fitting PCA on test data would make the evaluation unfair.

## PCA

PCA showed that 63 components preserve 90% of the variance, and 102 components preserve 95% of the variance.

The first two components were used for visualization, but they were not enough for the best classification performance.

## Models

I trained three models:

- Logistic Regression
- SVM with RBF kernel
- Random Forest

I trained them on the original features and also on PCA-reduced data.

## Results

The best model was Logistic Regression using the original 561 standardized features.

It achieved 95.45% accuracy and 95.44% macro F1-score.

The best PCA result was SVM with 150 components, with 94.71% accuracy.

This means PCA reduced the number of dimensions and kept strong performance, but the original features still performed best.

## Confusion Matrix

The main confusion was between sitting and standing.

The best model predicted sitting as standing 58 times, and standing as sitting 14 times.

This makes sense because both are static activities and produce similar sensor signals.

Some confusion also happened between walking, walking upstairs, and walking downstairs because they are similar dynamic activities.

## Conclusion

The project shows that smartphone sensor features can recognize human activity with high accuracy.

The best model reached about 95% performance.

PCA was useful for understanding and reducing the data, but the original engineered features gave the best result.

