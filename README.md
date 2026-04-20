# Customer Churn Prediction
A machine learning project that predicts whether a customer will churn (leave a service) based on behavioral and demographic features. The project includes extensive exploratory data analysis, multi-model experimentation, hyperparameter tuning, and a Flask web application for live predictions.
---
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Models Explored](#models-explored)
- [Final Model & Results](#final-model--results)
- [Web Application](#web-application)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
---
## Overview
Customer churn is a critical business metric. This project builds a binary classifier to identify customers likely to cancel their subscription, enabling proactive retention strategies. The workflow covers:
1. Data wrangling and merging of training/testing splits
2. Feature engineering (combining `Subscription Type` and `Contract Length` into a single `Subscription_Contract` feature)
3. Experimenting with multiple classification algorithms
4. Handling class imbalance via SMOTE
5. Hyperparameter tuning with Grid Search
6. Model evaluation with multiple metrics
7. Deployment as a Flask web app
---
## Dataset
- **Source files**: `customer_churn_dataset-training-master.csv` and `customer_churn_dataset-testing-master.csv`
- **Size**: ~505,206 records after merging and removing nulls
- **Target column**: `Churn` (0 = No Churn, 1 = Churn)
> The original train/test split exhibited poor stratification and significant class imbalance, so the two files were merged and re-split with stratification.
---
## Features
| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Customer age |
| Gender | Categorical | Male / Female |
| Tenure | Numerical | Months as a customer |
| Usage Frequency | Numerical | Product/service usage frequency |
| Support Calls | Numerical | Number of support calls made |
| Payment Delay | Numerical | Average payment delay (days) |
| Total Spend | Numerical | Total amount spent |
| Last Interaction | Numerical | Days since last interaction |
| Subscription Type | Categorical | Basic / Standard / Premium (encoded) |
| Contract Length | Categorical | Monthly / Quarterly / Annual (encoded) |
| **Subscription_Contract** | **Engineered** | Combined feature: `Subscription Type + (Contract Length − 1)²` |
`Subscription Type` and `Contract Length` are merged into the engineered feature `Subscription_Contract` and their originals are dropped before training.
---
## Project Structure
```
Customer-Churn-Prediction/
│
├── Root.ipynb                        # EDA, baseline models, and initial experiments
├── FINAL_RF_MODEL.ipynb              # Final Random Forest pipeline
├── Random_Forest_Grid_Search.ipynb   # Hyperparameter tuning with GridSearchCV
├── NNClassifier.ipynb                # Neural Network classifier (TensorFlow/Keras)
├── xgb_classifier.ipynb              # XGBoost classifier experiments
├── Smote_test.ipynb                  # Class imbalance handling with SMOTE
├── improve_acc_attempt.ipynb         # Advanced feature engineering experiments
├── DETAILED_EVALUATION_FINALE.ipynb  # Final model evaluation and metric plots
│
├── app.py                            # Flask web application
└── README.md
```
---
## Models Explored
| Model | Notes |
|---|---|
| Random Forest | **Final model** — best overall performance |
| XGBoost | Competitive alternative |
| Neural Network (Keras) | Deep learning approach via scikeras wrapper |
| K-Nearest Neighbors | Baseline non-parametric model |
| Naive Bayes (Gaussian) | Probabilistic baseline |
| Support Vector Machine | Kernel-based classifier |
| Decision Tree | Interpretable tree model |
| AdaBoost | Boosting ensemble |
| Voting Classifier | Ensemble of multiple models |
| Stacking Classifier | Meta-learning ensemble |
Class imbalance was addressed using **SMOTE** (Synthetic Minority Over-sampling Technique) within an `imblearn` Pipeline.
---
## Final Model & Results
The final model is a **Random Forest Classifier** tuned with **Grid Search**, trained inside a `sklearn` Pipeline that handles preprocessing automatically.
### Hyperparameters
```python
RandomForestClassifier(
    n_estimators=500,
    bootstrap=False,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=10
)
```
### Preprocessing Pipeline
- **Numerical features**: `StandardScaler`
- **Categorical features**: `OneHotEncoder` (for `Gender`)
- **Engineered feature**: `Subscription_Contract` treated as numerical
### Performance Metrics
| Metric | Score |
|---|---|
| **Accuracy** | **0.94** |
| **Precision** | **0.90** |
| **Recall** | **1.00** |
| **F1 Score** | **0.95** |
The trained model is serialized to `RF_MODEL_FINALE.pkl` using `pickle` for deployment.
---
## Web Application
A **Flask** web app (`app.py`) lets users input customer data through a form and receive a real-time churn prediction.
### Routes
| Route | Method | Description |
|---|---|---|
| `/` | GET | Renders the input form (`index.html`) |
| `/predict` | POST | Accepts form data, runs the model, returns the prediction |
### Prediction Output
- **Churn predicted**: Returns a thematic message and a visual indicator (`willchurn.jpg`)
- **No churn predicted**: Returns a positive retention message and visual (`wontchurn.jpg`)
---
## Tech Stack
| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Boosting | `xgboost` |
| Deep Learning | `tensorflow`, `keras`, `scikeras` |
| Imbalanced Learning | `imbalanced-learn` (SMOTE) |
| Web Framework | `Flask` |
| Model Serialization | `pickle` |
---
## Getting Started
### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost tensorflow scikeras imbalanced-learn flask matplotlib seaborn
```
### Running the Web App
1. Ensure `RF_MODEL_FINALE.pkl` is available and update the model path in `app.py`.
2. Start the Flask server:
```bash
python app.py
```
3. Open your browser and navigate to `http://127.0.0.1:5000/`.
### Running the Notebooks
Open any notebook in Jupyter and run the cells in order. The recommended sequence is:
1. `Root.ipynb` — EDA and baseline experiments
2. `Smote_test.ipynb` — Class imbalance handling
3. `Random_Forest_Grid_Search.ipynb` — Hyperparameter tuning
4. `FINAL_RF_MODEL.ipynb` — Final model training and export
5. `DETAILED_EVALUATION_FINALE.ipynb` — Full evaluation report
---
## License
This project is open source and available for educational and research purposes.
