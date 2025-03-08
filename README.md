# Breast Cancer Classification Project

## Project Overview

### Description
This project implements and compares various machine learning models for breast cancer classification using the Wisconsin Breast Cancer dataset. The goal is to accurately classify breast masses as either benign or malignant based on 30 numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses.

### Problem Statement
Early detection and accurate diagnosis of breast cancer are crucial for patient survival. This project addresses the challenge of automating the classification of breast masses using machine learning techniques to assist medical professionals in making more accurate diagnoses.

### Objectives
- Implement and compare multiple machine learning models for breast cancer classification
- Evaluate model performance using various metrics
- Identify the most effective model for practical applications
- Document challenges and solutions in model implementation

## Models Implemented

### Model Performance Summary
1. **Bernoulli Naive Bayes** (Best Performer)
   - Accuracy: 0.982
   - F1-Score: 0.986
   - AUC: 1.000

2. **Gaussian Naive Bayes**
   - Accuracy: 0.965
   - F1-Score: 0.972
   - AUC: 0.997

3. **Logistic Regression**
   - Accuracy: 0.965
   - F1-Score: 0.972
   - AUC: 0.997

4. **Linear Regression**
   - Accuracy: 0.956
   - F1-Score: 0.966
   - AUC: 0.992

5. **K-Nearest Neighbors (KNN)**
   - Accuracy: 0.956
   - F1-Score: 0.965
   - AUC: 0.981

6. **Support Vector Machine (SVM)**
   - Accuracy: 0.956
   - F1-Score: 0.965
   - AUC: 0.995



## Installation & Setup

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

### Environment Setup
1. Clone the repository
2. Install dependencies
3. Ensure Python 3.7+ is installed

## Results & Findings

### Best Performing Model
- **Bernoulli Naive Bayes** achieved the highest accuracy (0.982) and AUC (1.000)
- Excellent balance between performance and computational efficiency

### Key Insights
1. All models performed well with accuracies above 95%
2. Naive Bayes variants showed surprisingly strong performance
3. Model training times were minimal across all algorithms

### Visualization Descriptions
- `all_models_comparison.png`: Bar plot comparing accuracy, precision, recall, F1-score, and AUC across models
- `all_models_roc_comparison.png`: ROC curves for all models showing true positive vs false positive rates

## Acknowledgments

### Data Source
- UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Dataset
- Dr. William H. Wolberg, University of Wisconsin Hospitals

### References
1. Breast Cancer Wisconsin (Diagnostic) Data Set
   - UCI Machine Learning Repository
   - https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

2. Scikit-learn Documentation
   - https://scikit-learn.org/stable/

3. Feature Computation Methods
   - Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993)
   - Nuclear feature extraction for breast tumor diagnosis
