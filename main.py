import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_salary_data
from eda import (
    print_statistical_summary, plot_salary_distribution,
    plot_correlation_matrix, plot_salary_by_categories
)
from preprocessing import split_data
from train import build_model_pipelines, train_all_models
from evaluate import evaluate_models, print_evaluation_table
from tuner import tune_best_model, print_tuning_results
from inference import save_model, predict_salary

def section(title):
    print(f"\n{'-'*50}\n {title}\n{'-'*50}")

def main():
    start = time.perf_counter()

    # Step 1: Download the raw standard Adult dataset and engineer it into 
    # Indian Rupees based on salaries, education bonuses, and working hours.
    section("Phase 1: Load Data")
    df = load_salary_data(random_state=42)
    print(f"Dataset shape: {df.shape}")

    # Step 2: Exploratory Data Analysis (EDA). This builds all the charts
    # (histograms, boxplots, correlation heatmaps) and saves them in /plots.
    section("Phase 2: EDA")
    print_statistical_summary(df)
    plot_salary_distribution(df)
    plot_correlation_matrix(df)
    plot_salary_by_categories(df)

    # Step 3: Split the dataset into 80% training data and 20% unseen test data.
    # The preprocessor (scaling + encoding) is fitted ONLY on the training data.
    section("Phase 3: Preprocessing & Split")
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 4: Fit our baseline Linear Regression, Random Forest, and Gradient Boosting.
    section("Phase 4: Training Models")
    pipelines = build_model_pipelines()
    trained = train_all_models(pipelines, X_train, y_train)

    # Step 5: Test all models on the 20% unseen data. Calculate MAE, RMSE, and R2.
    # The model with the lowest Error (RMSE) wins.
    section("Phase 5: Evaluation")
    results, best_model = evaluate_models(trained, X_test, y_test)
    print_evaluation_table(results, best_model)

    # Step 6: Hyperparameter Tuning. We take the winning model and fine-tune its
    # internal settings (like tree depth) using Randomized Search to squeeze out extra accuracy.
    section(f"Phase 6: Tuning {best_model}")
    tuned_pipe, best_params, cv_rmse = tune_best_model(best_model, X_train, y_train, n_iter=10)
    print_tuning_results(best_model, best_params, cv_rmse)

    y_pred = tuned_pipe.predict(X_test)
    print("Test Performance (Unseen Data):")
    print(f"Mean Absolute Error (MAE)      : ₹{mean_absolute_error(y_test, y_pred):,.0f} (Avg prediction mistake)")
    print(f"Root Mean Squared Error (RMSE) : ₹{np.sqrt(mean_squared_error(y_test, y_pred)):,.0f} (Penalizes very wrong predictions)")
    print(f"R-Squared (R²)                 : {r2_score(y_test, y_pred):.4f} (1.0 is perfect accuracy)\n")

    section("Phase 7: Serialize Model")
    save_model(tuned_pipe)

    sample = {
        "Age": 35,
        "Years_of_Experience": 10,
        "Education_Level": "Master",
        "Job_Role": "Senior",
        "City_Tier": "Tier 1",
    }
    pred = predict_salary(sample)
    print(f"Sample: {sample}")
    print(f"Predicted Salary: ₹{pred:,.0f}\n")

    print("=" * 50)
    print(f"Pipeline finished in {time.perf_counter() - start:.1f}s")
    print("=" * 50)

if __name__ == "__main__":
    main()
