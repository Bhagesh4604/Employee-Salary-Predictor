import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter threading errors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = "plots"

def _rupee_formatter(x, _):
    return f"₹{x/1_00_000:.0f}L"

def _ensure_plot_dir(output_dir=PLOT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def print_statistical_summary(df):
    print("\nDataset Overview")
    print(f"Rows: {len(df):,}, Columns: {df.shape[1]}")
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nNumerical Summary:\n", df.describe(include=[np.number]).T)
    print("\nCategorical Summary:\n", df.describe(include=["object"]).T)

def plot_salary_distribution(df, output_dir=PLOT_DIR, show=False):
    _ensure_plot_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["Salary"], kde=True, bins=40, color="#4C72B0", ax=axes[0])
    axes[0].set_title("Annual Salary Distribution")
    axes[0].set_xlabel("Salary (₹)")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(_rupee_formatter))

    sns.histplot(np.log1p(df["Salary"]), kde=True, bins=40, color="#55A868", ax=axes[1])
    axes[1].set_title("Log Salary")

    plt.tight_layout()
    path = os.path.join(output_dir, "salary_distribution.png")
    fig.savefig(path, dpi=150)
    if show: plt.show()
    plt.close(fig)
    return path

def plot_correlation_matrix(df, output_dir=PLOT_DIR, show=False):
    _ensure_plot_dir(output_dir)
    corr = df.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    
    path = os.path.join(output_dir, "correlation_matrix.png")
    fig.savefig(path, dpi=150)
    if show: plt.show()
    plt.close(fig)
    return path

def plot_salary_by_categories(df, output_dir=PLOT_DIR, show=False):
    _ensure_plot_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=df, x="Education_Level", y="Salary", ax=axes[0], palette="Blues")
    axes[0].set_title("Salary by Education")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_rupee_formatter))

    sns.boxplot(data=df, x="Job_Role", y="Salary", ax=axes[1], palette="Oranges")
    axes[1].set_title("Salary by Role")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_rupee_formatter))

    sns.boxplot(data=df, x="City_Tier", y="Salary", ax=axes[2], palette="Greens")
    axes[2].set_title("Salary by City Tier")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(_rupee_formatter))

    plt.tight_layout()
    path = os.path.join(output_dir, "salary_by_categories.png")
    fig.savefig(path, dpi=150)
    if show: plt.show()
    plt.close(fig)
    return path

if __name__ == "__main__":
    from data_loader import load_salary_data
    df = load_salary_data()
    plot_salary_distribution(df)
    plot_correlation_matrix(df)
    plot_salary_by_categories(df)
