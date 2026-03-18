import pandas as pd
import yaml
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score


def load_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model():
    config = load_config()

    # -----------------------------
    # 1. Load Processed Data
    # -----------------------------
    processed_path = os.path.join(
        os.path.dirname(config["data"]["clean_data_path"]),
        "processed.csv"
    )

    df = pd.read_csv(processed_path)

    # -----------------------------
    # 2. Split Data
    # -----------------------------
    target_col = "Test Results"

    X = df.drop(columns=target_col)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. Train Model
    # -----------------------------
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # -----------------------------
    # 4. Predictions
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # 5. Evaluation
    # -----------------------------
    Accuracy_score = accuracy_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average="weighted")
    Confusion_matrix = confusion_matrix(y_test, y_pred)

    print("📊 Model Performance:")
    print(f"Accuracy: {Accuracy_score}")
    print(f"F1 Score: {F1_score}")
    print(f"Confusion Matrix: {Confusion_matrix}")

    # -----------------------------
    # 6. Save Model (FIXED PART)
    # -----------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)   # ✅ SAVE ONLY

    print(f"✅ Model saved at: {model_path}")


if __name__ == "__main__":
    train_model()