import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


# 1. Load and Clean Dataset

data = pd.read_csv("data/heart.csv")

print("Original Shape:", data.shape)

# Remove duplicates (CRITICAL)
data = data.drop_duplicates()

print("Shape After Removing Duplicates:", data.shape)
print("Duplicate Rows Removed:", data.duplicated().sum())


# 2. Separate Features and Target

X = data.drop("target", axis=1)
y = data["target"].astype(int)

print("\nClass Distribution:")
print(y.value_counts())


# 3. Feature Groups

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]


# 4. Preprocessing Pipeline

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", "passthrough", categorical_features)
    ]
)


# 5. Model Definition (Tuned XGBoost)

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", xgb_model)
])


# 6. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# 7. Cross Validation (5-Fold)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print("\nCross-Validation ROC-AUC: {:.4f} ± {:.4f}".format(
    cv_scores.mean(), cv_scores.std()
))


# 8. Train Final Model

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n===== Final Test Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 9. SHAP Explainability

# Extract trained XGBoost model
trained_xgb = model.named_steps["clf"]

# Transform full dataset (without leakage)
X_processed = model.named_steps["preprocess"].transform(X)

# SHAP explainer
explainer = shap.TreeExplainer(trained_xgb)
shap_values = explainer.shap_values(X_processed)

# Summary Plot (Bar)
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("models/shap_feature_importance.png")
plt.close()

print("\nSHAP feature importance plot saved as models/shap_feature_importance.png")


# 10. Save Model

joblib.dump(model, "models/final_heart_model.pkl")

print("\nModel saved successfully!")


# 11. Feature Importance Ranking

# Get feature names after preprocessing
feature_names = (
    model.named_steps["preprocess"]
    .get_feature_names_out()
)

# Get feature importance from XGBoost
importances = model.named_steps["clf"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n===== Feature Importance Ranking =====")
print(importance_df)

# Save to CSV
importance_df.to_csv("models/feature_importance.csv", index=False)

print("\nFeature importance table saved as models/feature_importance.csv")


# 12. Probability Threshold Tuning


threshold = 0.40  # Try lowering from default 0.5

y_custom_pred = (y_proba >= threshold).astype(int)

print(f"\n===== Custom Threshold Evaluation (Threshold = {threshold}) =====")
print("Accuracy:", accuracy_score(y_test, y_custom_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_custom_pred))
print("\nClassification Report:\n", classification_report(y_test, y_custom_pred))