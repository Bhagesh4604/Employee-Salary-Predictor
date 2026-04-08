import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(trained_pipelines, X_test, y_test):
    records = []
    
    for name, pipe in trained_pipelines.items():
        y_pred = pipe.predict(X_test)
        
        # Calculate evaluation metrics to see how well the model performed:
        # MAE: average error in rupees
        mae = mean_absolute_error(y_test, y_pred)
        # RMSE: penalizes large errors more heavily than small ones
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # R2 Score: percentage of variance explained by the model (closer to 1 is better)
        r2 = r2_score(y_test, y_pred)
        
        records.append({
            "Model": name,
            "Mean Absolute Error (MAE)": round(mae, 2),
            "Root Mean Sq Error (RMSE)": round(rmse, 2),
            "R-Squared (R²)": round(r2, 4),
        })

    results_df = pd.DataFrame(records).set_index("Model").sort_values("Root Mean Sq Error (RMSE)")
    best_model_name = results_df.index[0]
    return results_df, best_model_name

def print_evaluation_table(results_df, best_model_name):
    print("\nModel Comparison")
    
    display = results_df.copy()
    display["Mean Absolute Error (MAE)"] = display["Mean Absolute Error (MAE)"].map("₹{:,.2f}".format)
    display["Root Mean Sq Error (RMSE)"] = display["Root Mean Sq Error (RMSE)"].map("₹{:,.2f}".format)
    display["R-Squared (R²)"] = display["R-Squared (R²)"].map("{:.4f}".format)

    print(display.to_string())
    print("-" * 40)
    print(f"Best Model: {best_model_name}")

if __name__ == "__main__":
    from data_loader import load_salary_data
    from preprocessing import split_data
    from train import build_model_pipelines, train_all_models

    df = load_salary_data()
    X_train, X_test, y_train, y_test = split_data(df)
    trained = train_all_models(build_model_pipelines(), X_train, y_train)
    results_df, best = evaluate_models(trained, X_test, y_test)
    print_evaluation_table(results_df, best)
