import os
import joblib
import pandas as pd
import numpy as np
from preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

DEFAULT_MODEL_PATH = "salary_model.joblib"

def save_model(pipeline, path=DEFAULT_MODEL_PATH):
    joblib.dump(pipeline, path, compress=3)
    print(f"Model saved to {path}")

def load_model(path=DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def predict_salary(employee_data):
    row_df = pd.DataFrame([employee_data])
    for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        if col not in row_df.columns:
            raise ValueError(f"Missing column: {col}")
    
    pipeline = load_model(DEFAULT_MODEL_PATH)
    pred = pipeline.predict(row_df)
    return round(float(pred[0]), 0)

if __name__ == "__main__":
    sample = {
        "Age": 28,
        "Years_of_Experience": 5,
        "Education_Level": "Bachelor",
        "Job_Role": "Senior",
        "City_Tier": "Tier 1",
    }
    try:
        salary = predict_salary(sample)
        print(f"Predicted Salary: ₹{salary:,.0f}")
    except FileNotFoundError:
        print("Run main.py to train model first.")
