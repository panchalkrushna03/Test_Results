import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os

# -----------------------------
# 1. Load Model (IMPORTANT)
# -----------------------------


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")


model = joblib.load(model_path)

# -----------------------------
# 2. Create FastAPI App
# -----------------------------
app = FastAPI()


# -----------------------------
# 3. Input Schema
# -----------------------------
class InputData(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Insurance_Provider: str
    Admission_Type: str
    Medication: str


# -----------------------------
# 4. Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Healthcare Prediction API Running 🚀"}


# -----------------------------
# 5. Prediction Route
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to dictionary
        input_dict = data.model_dump()

        # Convert to DataFrame
        df = pd.DataFrame([input_dict])

        # 🔥 Fix column names (important for your dataset)
        df.columns = df.columns.str.replace("_", " ")

        # Prediction
        prediction = model.predict(df)[0]

        return {
            "Predicted Billing Amount": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}