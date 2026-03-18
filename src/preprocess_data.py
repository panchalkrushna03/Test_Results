import pandas as pd
import yaml
import os
import joblib
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_data():
    config = load_config()

    # -----------------------------
    # 1. Load Data
    # -----------------------------
    df = pd.read_csv(config["data"]["raw_data_path"])

    # 🔥 Fix column name issues
    df.columns = df.columns.str.strip()

    # -----------------------------
    # 2. Drop Columns
    # -----------------------------
    columns_to_drop = [
        'Name', 'Date of Admission', 'Discharge Date',
        'Doctor', 'Hospital', 'Room Number'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    # -----------------------------
    # 3. Save Clean Data
    # -----------------------------
    clean_path = config["data"]["clean_data_path"]
    dir_path = os.path.dirname(clean_path)

    os.makedirs(dir_path, exist_ok=True)

    df.to_csv(clean_path, index=False)
    print("✅ Cleaned data saved")

    # -----------------------------
    # 4. Encoding + Scaling
    # -----------------------------
    target_col = "Test Results"   # ✅ your target
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    X = df.drop(columns=target_col)
    y = df[target_col]

    # Column types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

    # Transform
    X_processed = preprocessor.fit_transform(X)

    # Column names
    if len(cat_cols) > 0:
        ohe_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        all_cols = list(ohe_cols) + num_cols
    else:
        all_cols = num_cols

    # Convert to DataFrame
    X_df = pd.DataFrame(X_processed, columns=all_cols)

    # Add target back
    final_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    # -----------------------------
    # 5. Save Processed Data (same folder)
    # -----------------------------
    processed_path = os.path.join(dir_path, "processed.csv")

    final_df.to_csv(processed_path, index=False)

    print(f"✅ Processed data saved at: {processed_path}")

    # -----------------------------
    # 6. Save Preprocessor
    # -----------------------------
    preprocessor_path = config["artifacts"]["preprocessor_path"]

    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    joblib.dump(preprocessor, preprocessor_path)

    print("✅ Preprocessor saved")


if __name__ == "__main__":
    clean_data()