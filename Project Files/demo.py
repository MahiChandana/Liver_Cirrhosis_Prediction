# demo.py

# Step 0: Environment Setup
import sys
print("Running with Python executable:", sys.executable)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Step 2: Load Excel File
file_path = "HealthCareData.xlsx"
df = pd.read_excel(file_path)

# Step 3: Display Columns
print("\n✅ Available columns in the dataset:\n", df.columns.tolist())

# Step 4: Clean column names (remove spaces etc.)
df.columns = df.columns.str.strip()

# Step 5: Convert specific known numeric columns (optional)
cols_to_convert = [col for col in ['TG', 'LDL', 'Total Bilirubin    (mg/dl)', 'A/G Ratio'] if col in df.columns]
if cols_to_convert:
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Step 6: Fill numeric missing values
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Step 7: Detect and fill categorical columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 8: Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ✅ Step 9: Automatically detect target column
possible_targets = [col for col in df.columns if 'predict' in col.lower() or 'outcome' in col.lower() or 'cirrosis' in col.lower()]
if not possible_targets:
    raise ValueError("❌ Target column not found automatically. Please check column names.")
target = possible_targets[0]
print(f"\n🎯 Target column detected as: '{target}'")

# Step 10: Define Features and Target
X = df.drop(columns=[target])
y = df[target].astype(int) if df[target].dtype != 'int' else df[target]

# Step 11: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 12: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 13: Train and Evaluate Models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Logistic Regression CV": LogisticRegressionCV(cv=5, max_iter=1000),
        "Ridge Classifier": RidgeClassifier(),
        "KNN Classifier": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n🔍 {name}")
        print("-" * 40)
        print("Train Accuracy:", model.score(X_train, y_train))
        print("Test Accuracy:", model.score(X_test, y_test))
        print("Precision:", precision_score(y_test, y_pred, average='weighted'))
        print("Recall:", recall_score(y_test, y_pred, average='weighted'))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        results[name] = {
            "model": model,
            "train_accuracy": model.score(X_train, y_train),
            "test_accuracy": model.score(X_test, y_test),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }

    return results

model_results = evaluate_models(X_train, X_test, y_train, y_test)

# Step 14: Feature Importance (Random Forest)
rf_model = model_results["Random Forest"]["model"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# Step 15: Save Best Model and Tools
best_model_name = max(model_results, key=lambda x: model_results[x]['test_accuracy'])
best_model = model_results[best_model_name]["model"]

with open("best_model.pkl", "wb") as file:
    pickle.dump((best_model, scaler, label_encoders, X.columns.tolist()), file)

print(f"\n✅ Best model '{best_model_name}' and tools saved successfully as 'best_model.pkl'")