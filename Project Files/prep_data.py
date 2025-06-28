import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# ✅ Show Python environment info
print("Running with Python executable:", sys.executable)

# 📁 Load dataset
file_path = 'HealthCareData.xlsx'  # Update this path as needed
df = pd.read_excel(file_path)

# 🧹 Clean column names
df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=True).str.replace('\r', '', regex=True)
print("📋 Columns found:", df.columns.tolist())

# 🔢 Convert numeric candidates
numeric_columns = ['TG', 'LDL', 'Total_Bilirubin_mgdl', 'AG_Ratio']
valid_numeric_cols = [col for col in numeric_columns if col in df.columns]
df[valid_numeric_cols] = df[valid_numeric_cols].apply(pd.to_numeric, errors='coerce')

# 🧼 Fill missing values
# Fill numeric with median
num_cols = df.select_dtypes(include='number').columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print(f"⚠️ No mode for column '{col}'. Check its values.")

# 🎯 Detect target column
target_name = 'Predicted ValueOut_ComePatient suffering from liver cirrosis or not'
if target_name not in df.columns:
    matches = [col for col in df.columns if 'cirrosis' in col.lower()]
    if matches:
        target_name = matches[0]
        print(f"✅ Detected target column: '{target_name}'")
    else:
        print("❌ Target column not found.")
        target_name = None
else:
    print(f"✅ Using target column: '{target_name}'")

# 🧠 Encode target
if target_name:
    df = df[df[target_name].notna()]
    if df[target_name].dtype == 'object':
        target_encoder = LabelEncoder()
        df[target_name] = target_encoder.fit_transform(df[target_name].astype(str))
        joblib.dump(target_encoder, 'target_encoder.pkl')

# 🧬 Encode categorical features
encoders = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 💾 Save encoders
joblib.dump(encoders, 'label_encoders.pkl')

# ✅ Done
print("✅ Preprocessing complete. Final dataset shape:", df.shape)
