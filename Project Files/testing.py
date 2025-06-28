# performance_testing.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load data
df = pd.read_excel('HealthCareData.xlsx', sheet_name='Sheet1')

# 2️⃣ Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r'\s+', '', regex=True)
    .str.replace(r'[^A-Za-z0-9]+', '', regex=True)
)

# 3️⃣ Identify and rename target column
print("\n📋 Cleaned Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

target_col = next((col for col in df.columns if 'cirrhosis' in col.lower() or 'outcome' in col.lower()), None)

if target_col:
    df.rename(columns={target_col: 'Target'}, inplace=True)
    print(f"\n✅ Target column set as: '{target_col}'")
else:
    raise ValueError("❌ Could not identify the target column. Please check the column names above.")

# 4️⃣ Remove rows with missing target
df.dropna(subset=['Target'], inplace=True)

# 5️⃣ Encode target values (YES/NO → 1/0)
df['Target'] = df['Target'].astype(str).str.upper().map({'YES': 1, 'NO': 0})
df.dropna(subset=['Target'], inplace=True)

# 6️⃣ Drop unnecessary identifier columns
if 'SNO' in df.columns:
    df.drop(columns='SNO', inplace=True)

# 7️⃣ Encode categorical features
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 8️⃣ Split data
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9️⃣ Train base model
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)

# 🔟 Evaluate base model
print("\n📊 Base Model Evaluation:")
print(classification_report(y_test, y_pred))
print("🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📈 ROC AUC Score:", roc_auc_score(y_test, y_pred))

# 1️⃣1️⃣ Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 1️⃣2️⃣ Evaluate tuned model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n✅ Best Parameters:", grid_search.best_params_)
print("\n📌 Tuned Model Evaluation:")
print(classification_report(y_test, y_pred_best))
print("🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("📈 ROC AUC Score:", roc_auc_score(y_test, y_pred_best))
