# data_loader.py
# Handles downloading the UCI Adult dataset and organizing the columns.
# We convert the binary income classes into a continuous salary feature in INR
# by factoring in education, job role, and hours worked.

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

CACHE_PATH = "data/salary_processed.csv"

EDUCATION_MAP = {
    "Preschool": "High School", "1st-4th": "High School",
    "5th-6th": "High School", "7th-8th": "High School",
    "9th": "High School", "10th": "High School",
    "11th": "High School", "12th": "High School",
    "HS-grad": "High School", "Some-college": "Bachelor",
    "Assoc-voc": "Bachelor", "Assoc-acdm": "Bachelor",
    "Bachelors": "Bachelor", "Prof-school": "Master",
    "Masters": "Master", "Doctorate": "PhD",
}

OCCUPATION_MAP = {
    "Priv-house-serv": "Junior", "Other-service": "Junior",
    "Handlers-cleaners": "Junior", "Farming-fishing": "Junior",
    "Machine-op-inspct": "Junior", "Adm-clerical": "Mid",
    "Sales": "Mid", "Transport-moving": "Mid",
    "Craft-repair": "Mid", "Tech-support": "Senior",
    "Protective-serv": "Senior", "Armed-Forces": "Senior",
    "Prof-specialty": "Lead", "Exec-managerial": "Manager",
}

BASE_SALARY_INR = {">50K": 10_00_000, "<=50K": 4_00_000}

EDUCATION_BONUS = {
    "High School": 0,
    "Bachelor": 1_50_000,
    "Master": 3_00_000,
    "PhD": 5_00_000,
}

ROLE_BONUS = {
    "Junior": 0,
    "Mid": 2_00_000,
    "Senior": 5_00_000,
    "Lead": 9_00_000,
    "Manager": 15_00_000,
}

TIER_PROBS = {
    "Manager": [0.65, 0.25, 0.10],
    "Lead": [0.55, 0.30, 0.15],
    "Senior": [0.45, 0.35, 0.20],
    "Mid": [0.30, 0.45, 0.25],
    "Junior": [0.20, 0.40, 0.40],
}
TIERS = ["Tier 1", "Tier 2", "Tier 3"]

def load_salary_data(random_state=42):
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    print("Downloading Adult dataset...")
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    raw = data.frame.copy()

    rng = np.random.default_rng(random_state)
    key_cols = ["age", "education", "occupation", "hours-per-week", "class"]
    raw.dropna(subset=key_cols, inplace=True)

    raw["Education_Level"] = raw["education"].map(EDUCATION_MAP).fillna("Bachelor")
    raw["Job_Role"] = raw["occupation"].map(OCCUPATION_MAP).fillna("Mid")
    raw["Age"] = pd.to_numeric(raw["age"], errors="coerce")

    exp_noise = rng.normal(0, 2.0, len(raw))
    raw["Years_of_Experience"] = np.clip(
        (raw["Age"] - 22) + exp_noise, 0, raw["Age"] - 18
    ).round(1)

    income_class = raw["class"].astype(str).str.strip()
    base = np.where(income_class.str.contains(">50K"), 10_00_000, 4_00_000).astype(float)

    hours = pd.to_numeric(raw["hours-per-week"], errors="coerce").fillna(40)
    overtime = np.clip(hours - 40, 0, None).values * 1_500

    edu_bonus = raw["Education_Level"].map(EDUCATION_BONUS).fillna(0).values
    role_bonus = raw["Job_Role"].map(ROLE_BONUS).fillna(0).values
    noise = rng.normal(0, 1_20_000, len(raw))

    salary = base + overtime + edu_bonus + role_bonus + noise
    salary = np.maximum(salary, 1_80_000)
    raw["Salary"] = np.round(salary, 0)

    raw["City_Tier"] = [
        rng.choice(TIERS, p=TIER_PROBS.get(role, [0.33, 0.34, 0.33]))
        for role in raw["Job_Role"]
    ]

    df = raw[[
        "Age", "Years_of_Experience", "Education_Level", 
        "Job_Role", "City_Tier", "Salary"
    ]].dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    
    print("Dataset saved to cache.")
    return df

load_data = load_salary_data

if __name__ == "__main__":
    df = load_salary_data()
    print(df.head())
