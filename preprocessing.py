import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

NUMERICAL_FEATURES = ["Age", "Years_of_Experience"]
CATEGORICAL_FEATURES = ["Education_Level", "Job_Role", "City_Tier"]
TARGET = "Salary"

def build_preprocessor():
    # For numbers: replace missing values with the median (handles outliers better than mean),
    # then scale them so big numbers (like salary) don't dominate small numbers (like age).
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # For text categories: fill missing values with the most common category, 
    # then convert text into 1s and 0s (One-Hot Encoding) so the model can understand it.
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def split_data(df, test_size=0.20, random_state=42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from data_loader import load_salary_data
    df = load_salary_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    print(f"Processed train shape: {X_train_proc.shape}")
