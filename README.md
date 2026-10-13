# Industrial Machine Failure Prediction

## Project Overview
This project uses machine learning to predict industrial machine failures based on operating conditions such as temperature, rotational speed, torque, and tool wear.

The XGBoost model is deployed using Streamlit and Render.

## Dataset
- Dataset: AI4I 2020 Predictive Maintenance Dataset
- Total Records: 10,000
- Input Features: 6
- Target Variable: `Machine failure`
  - `0` – No Failure
  - `1` – Machine Failure

## Machine Learning Models
1. Decision Tree
2. Support Vector Machine (SVM)
3. XGBoost

**Best Model:** XGBoost

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.00% |
| Precision | 67.07% |
| Recall | 80.88% |
| F1-Score | 73.33% |
| ROC-AUC | 97.01% |

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Streamlit
- Render
- GitHub

## Deployment

**Live Application:**  
https://machine-failure-prediction1-sqgy.onrender.com

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
