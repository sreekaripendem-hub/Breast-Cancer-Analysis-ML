# Breast Cancer Analysis Using Machine Learning

## Overview

This project explores supervised and unsupervised machine learning techniques using the **Wisconsin Breast Cancer Dataset**. It focuses on predicting benign and malignant tumours through classification algorithms and identifying patterns in the dataset using clustering techniques.

The project investigates Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbours (KNN), and K-Means Clustering, along with model evaluation, hyperparameter tuning and dimensionality reduction.

## Dataset

The project uses the Wisconsin Breast Cancer Dataset.

* 569 observations
* 30 numerical features
* 2 diagnosis classes: Benign (B) and Malignant (M)
* 357 benign observations
* 212 malignant observations

The features include measurements related to radius, texture, perimeter, area, smoothness, compactness and concavity.

## Project Objectives

* Explore and preprocess the dataset.
* Analyse feature correlations with diagnosis.
* Understand Logistic Regression probabilities and log-odds.
* Compare single-feature and multi-feature classification.
* Evaluate Logistic Regression and KNN.
* Implement SVM with different kernel functions.
* Perform hyperparameter tuning using 5-Fold Cross-Validation.
* Apply K-Means Clustering.
* Determine the optimal number of clusters using Elbow and Silhouette Analysis.
* Visualize clustering results using PCA.

## Methodology

### 1. Data Preprocessing and Feature Analysis

* Loaded and explored the dataset using Pandas.
* Separated features and diagnosis labels.
* Converted diagnosis into binary labels.
* Performed stratified 80:20 train-test splitting.
* Standardized features using StandardScaler.
* Analysed feature correlations using Pearson correlation.

The feature with the highest absolute Pearson correlation was `concave points_worst`, with a correlation of approximately 0.7863.

### 2. Logistic Regression

* Implemented Logistic Regression for binary classification.
* Analysed predicted probabilities and log-odds.
* Verified the relationship between log-odds and predicted probabilities using the sigmoid function.
* Ranked features based on model coefficients.
* Compared single-feature and all-feature models.
* Used GridSearchCV with 5-Fold Cross-Validation for hyperparameter tuning.

### 3. Logistic Regression vs KNN

Compared Logistic Regression and KNN using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Log-Loss

### 4. Support Vector Machines (SVM)

Implemented and compared four SVM kernel functions:

* Linear Kernel
* Polynomial Kernel
* Radial Basis Function (RBF) Kernel
* Sigmoid Kernel

The analysis included predicted labels, decision scores, probabilities and hyperparameter tuning using Stratified 5-Fold Cross-Validation.

### 5. K-Means Clustering

Applied K-Means Clustering for different cluster counts:

* k = 2
* k = 3
* k = 4
* k = 5
* k = 6

Evaluated clustering performance using:

* Inertia
* Silhouette Score
* Elbow Method

The preferred number of clusters was **k = 2**, based on the highest silhouette score of 0.3434.

### 6. PCA Visualisation

* Applied Principal Component Analysis (PCA).
* Reduced the feature space to two principal components.
* Visualized the dataset and K-Means clusters.
* Compared cluster assignments with actual diagnosis labels.

The first two principal components explained approximately 63.24% of the total variance.

## Key Results

### Logistic Regression vs KNN

| Metric    | Logistic Regression |    KNN |
| --------- | ------------------: | -----: |
| Accuracy  |              96.49% | 94.74% |
| Precision |              97.50% | 97.37% |
| Recall    |              92.86% | 88.10% |
| F1-Score  |              95.12% | 92.50% |
| ROC-AUC   |              99.60% | 98.18% |

### SVM Kernel Comparison

| Kernel     | Accuracy | F1-Score |
| ---------- | -------: | -------: |
| Linear     |   97.37% |   96.30% |
| Polynomial |   87.72% |   80.00% |
| RBF        |   96.49% |   95.00% |
| Sigmoid    |   97.37% |   96.30% |

### K-Means Clustering

* Preferred cluster count: k = 2
* Highest silhouette score: 0.3434
* PCA dimensionality: 2 components
* Explained variance: 63.24%

The two clusters broadly corresponded to the benign and malignant groups, with some overlap.

## Technologies and Libraries

* Python
* Jupyter Notebook
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

