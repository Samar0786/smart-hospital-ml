# 🏥 Smart Hospital - AI Heart Disease Prediction System

A production-ready Machine Learning system for heart disease risk prediction using XGBoost with SHAP explainability.

---

## 📌 Project Overview

This project builds a clinical decision support model to predict heart disease risk using structured patient data.

The system:
- Cleans and preprocesses medical data
- Removes duplicate records
- Uses stratified cross-validation
- Trains an optimized XGBoost classifier
- Evaluates using ROC-AUC and Accuracy
- Provides SHAP-based model interpretability
- Exports feature importance ranking
- Supports probability threshold tuning

This model is designed for integration into a hospital management system backend.

---

## 🧠 Model Used

- XGBoost (Gradient Boosted Trees)
- 5-Fold Stratified Cross Validation
- Evaluation Metric: ROC-AUC
- SHAP for explainability

---

## 📊 Performance (Clean Dataset)

- Cross-Validation ROC-AUC: ~0.88 – 0.90
- Test Accuracy: ~85–90%
- Robust against overfitting
- No data leakage

---

## 🗂 Project Structure
smart-hospital-ml/
│
├── data/
│ └── heart.csv
│
├── models/
│ ├── final_heart_model.pkl
│ ├── shap_feature_importance.png
│ └── feature_importance.csv
│
├── train.py
├── requirements.txt
└── README.md


---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/smart-hospital-ml.git
cd smart-hospital-ml

Create virtual environment:

python -m venv venv
source venv/Scripts/activate  # Windows

Install dependencies:

pip install -r requirements.txt
▶️ Train Model
python train.py

This will:

Train model

Evaluate performance

Generate SHAP feature importance

Save trained model in models/

📈 Explainability

SHAP is used to:

Identify most important risk factors

Improve model transparency

Support clinical interpretability

⚙️ Deployment Ready

The model is saved as:

models/final_heart_model.pkl

It can be loaded directly into a FastAPI or Flask backend.