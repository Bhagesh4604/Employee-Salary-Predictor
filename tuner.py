import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from preprocessing import build_preprocessor

RFR_PARAM_DIST = {
    "regressor__n_estimators": [100, 200, 300, 500],
    "regressor__max_depth": [None, 5, 10, 15, 20],
    "regressor__max_features": ["sqrt", "log2", 0.5, 0.8],
    "regressor__min_samples_leaf": [1, 3, 5, 10],
    "regressor__min_samples_split": [2, 5, 10],
}

GBR_PARAM_DIST = {
    "regressor__n_estimators": [200, 300, 400, 500],
    "regressor__learning_rate": [0.01, 0.05, 0.10, 0.15],
    "regressor__max_depth": [3, 4, 5, 6],
    "regressor__subsample": [0.6, 0.7, 0.8, 1.0],
    "regressor__min_samples_leaf": [5, 10, 15, 20],
}

def tune_best_model(best_name, X_train, y_train, n_iter=30, cv=5):
    name_lower = best_name.lower()
    
    if "random forest" in name_lower:
        regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = RFR_PARAM_DIST
    elif "gradient boosting" in name_lower:
        regressor = GradientBoostingRegressor(random_state=42)
        param_dist = GBR_PARAM_DIST
    else:
        raise ValueError("Unsupported model for tuning")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", regressor)
    ])

    print(f"Tuning {best_name}...")
    
    # We use RandomizedSearchCV instead of testing every single combination (GridSearch).
    # It picks random combinations of hyperparameters and tests them using Cross Validation,
    # which is much faster and gives almost identical results.
    search = RandomizedSearchCV(
        estimator=pipeline, param_distributions=param_dist, n_iter=n_iter,
        scoring="neg_root_mean_squared_error", cv=cv, verbose=1, 
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, -search.best_score_

def print_tuning_results(model_name, best_params, best_rmse):
    print(f"\nTuning Results for {model_name}")
    print(f"Best RMSE: ₹{best_rmse:,.0f}")
    print("Best Params:")
    for k, v in best_params.items():
        print(f"  {k.replace('regressor__', '')}: {v}")

if __name__ == "__main__":
    from data_loader import load_salary_data
    from preprocessing import split_data
    df = load_salary_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipe, params, rmse = tune_best_model("Random Forest", X_train, y_train, n_iter=5)
    print_tuning_results("Random Forest", params, rmse)
