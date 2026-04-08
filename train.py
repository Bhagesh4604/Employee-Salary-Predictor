import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from preprocessing import build_preprocessor

def build_model_pipelines():
    # Linear Regression: A simple baseline model that tries to find a straight 
    # line relationship between experience/education and salary.
    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])

    # Random Forest: An ensemble method that builds multiple decision trees 
    # and averages their predictions. Great for capturing non-linear relationships 
    # without needing to manually engineer complex features.
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=200, max_features="sqrt", 
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )),
    ])

    # Gradient Boosting: Builds trees one by one, where each new tree tries 
    # to correct the errors of the previous ones. Usually gives the highest accuracy.
    gb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.10, max_depth=4, 
            subsample=0.80, min_samples_leaf=10, random_state=42
        )),
    ])

    return {
        "Linear Regression": lr_pipeline,
        "Random Forest": rf_pipeline,
        "Gradient Boosting": gb_pipeline,
    }

def train_all_models(pipelines, X_train, y_train):
    for name, pipe in pipelines.items():
        print(f"Fitting {name}...")
        pipe.fit(X_train, y_train)
    return pipelines

if __name__ == "__main__":
    from data_loader import load_salary_data
    from preprocessing import split_data
    df = load_salary_data()
    X_train, X_test, y_train, y_test = split_data(df)
    trained = train_all_models(build_model_pipelines(), X_train, y_train)
