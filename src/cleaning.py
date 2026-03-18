import pandas as pd
import yaml
import os
import joblib

from sklearn.preprocessing import LabelEncoder


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

    # Fix column names (important)
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
    # 3. Handle Missing Values
    # -----------------------------
    # ⚠️ Make sure correct column name
    target_col = "Test Results"

    df[target_col] = df[target_col].fillna(df[target_col].mode()[0])

    # -----------------------------
    # 🔥 4. Label Encoding (TARGET)
    # -----------------------------
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    # -----------------------------
    # 5. Save Clean Data
    # -----------------------------
    clean_path = config["data"]["clean_data_path"]
    dir_path = os.path.dirname(clean_path)

    os.makedirs(dir_path, exist_ok=True)

    df.to_csv(clean_path, index=False)

    print("✅ Cleaned + Encoded data saved")

    # -----------------------------
    # 6. Save Encoder (IMPORTANT)
    # -----------------------------
    encoder_path = config["artifacts"].get(
        "label_encoder_path", "artifacts/label_encoder.pkl"
    )

    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(le, encoder_path)

    print("✅ Label Encoder saved")


if __name__ == "__main__":
    clean_data()