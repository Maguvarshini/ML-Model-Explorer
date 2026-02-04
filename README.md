# ML Model Explorer 

## Overview
**ML Model Explorer** is an interactive web application built using **Streamlit** that allows users to explore, compare, and understand the behavior of different **machine learning classification models**.  
The app provides an intuitive interface to experiment with popular datasets, tune model hyperparameters in real time, and visualize model performance using multiple evaluation metrics.

 **For Live Demo:** [click here](http://localhost:8501/)

---

## Features

### Dataset Selection
Choose from classic machine learning datasets:
- Iris Dataset
- Breast Cancer Dataset
- Wine Dataset

---

### Supported Classifiers
The application supports multiple popular classification algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- Naive Bayes

---

### Interactive Model Tuning
- Real-time hyperparameter tuning using Streamlit sliders
- Classifier-specific parameter controls:
  - **Logistic Regression:** Regularization parameter (C)
  - **KNN:** Number of neighbors (K)
  - **SVM:** Regularization parameter (C)
  - **Decision Tree:** Maximum depth
  - **Random Forest:** Number of estimators, Maximum depth
  - **Gradient Boosting:** Number of estimators, Maximum depth

---

### Performance Analytics
Comprehensive evaluation metrics to analyze model performance:
- Accuracy Score
- Precision Score
- Recall Score
- F1 Score

#### Visualizations
- Interactive Confusion Matrix
- Detailed Classification Report
- ROC Curve (for binary classification problems)

---

## Tech Stack
- **Frontend & App Framework:** Streamlit  
- **Machine Learning:** scikit-learn  
- **Data Handling:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  

---

## Usage
1. Select a dataset from the sidebar dropdown menu.
2. Choose a machine learning classifier.
3. Adjust the model hyperparameters using the interactive sliders.
4. Click the **"Predict"** button to train and evaluate the model.
5. Analyze the model performance using the displayed metrics and visualizations.




