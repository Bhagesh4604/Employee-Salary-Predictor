# Employee Salary Predictor (Indian Edition) 💸

This project is a complete End-to-End Machine Learning pipeline designed to predict an employee's annual salary (in INR) based on their age, education, years of experience, job role, and city tier.

## 📊 1. Data Source & Approach
We built this model using the **UCI Adult Census Income Dataset**, which contains real-world demographic data of over 48,000 individuals. 

Since the original dataset classifies income as binary (`>50K` or `<=50K` USD), we "Indianized" the logic:
- We set base salaries (e.g., ₹10 Lakhs vs ₹4 Lakhs) based on the original income brackets.
- We added dynamic, realistic bonuses based on Education Level (e.g., +₹5 LPA for a PhD) and Job Role (e.g., +₹15 LPA for a Manager).
- We calculated realistic Years of Experience based on age minus educational years.
- Finally, added Gaussian "noise" to simulate unique individual differences in salary negotiations.

---

## 🛠️ 2. Project Steps & Phases

The codebase was modularized into the following distinct phases to simulate a professional data engineering environment:

1. **Data Ingestion (`data_loader.py`):** Automatically downloads the OpenML census data, transforms the categorical data, calculates experience, injects realistic INR logic, and caches a local `salary_processed.csv`.
2. **Exploratory Data Analysis (`eda.py`):** Automatically generates standard statistical summaries and saves correlation matrixes, histograms, and box plots into a `plots/` folder so we can understand our features visually before training.
3. **Preprocessing (`preprocessing.py`):** The data pipeline is prepared. Missing numerical values are imputed using the Median (handling outliers), and standardized (StandardScaler). Categorical strings (like "Master" or "Tier 1") are One-Hot Encoded mathematically into 1s and 0s. 
4. **Training (`train.py`):** Defines the machine learning pipelines and fits 80% of our data to train three distinct regression models.
5. **Evaluation (`evaluate.py`):** The models are tested on the remaining 20% "unseen" data points to see how accurate they truly are.
6. **Hyperparameter Tuning (`tuner.py`):** Takes the absolute best performing architecture from Phase 5 and randomly searches (`RandomizedSearchCV`) through hundreds of internal settings (hyperparameters) dynamically to squeeze out the absolute maximum amount of mathematical accuracy possible.
7. **Inference & UI (`inference.py` & `app.py`):** Saves the winning tuned pipeline as a compressed binary file (`salary_model.joblib`), then hosts a gorgeous **Streamlit** user interface that users can interact with!

---

## 🧠 3. Algorithms Used

Three algorithms were trained and compared in this project to capture complex relationships mapping demographics to salary:

1. **Linear Regression:**
   - **How it works:** Tries to draw a straight "line of best fit" through the data. It assumes that as experience strictly goes up, salary strictly goes up at a constant mathematical rate.
   - **Purpose:** Used as our "Baseline". It usually gets beaten, but if it doesn't, it means our dataset is remarkably simple. 
2. **Random Forest Regressor:**
   - **How it works:** An "Ensemble" algorithm. It builds hundreds of individual "Decision Trees". Each tree creates a complex flowchart (like "If Age > 30 and Role = Manager, Salary = X"). Then, the forest averages the predictions of all the trees together to give a highly robust output.
   - **Purpose:** Handles non-linear data incredibly well, preventing the model from just memorizing outliers (overfitting).
3. **Gradient Boosting Regressor:**
   - **How it works:** Also an ensemble of trees, but built sequentially. Tree #1 makes a guess. Tree #2 looks at where Tree #1 mathematically failed (the residuals) and tries to exclusively fix those mistakes. Tree #3 fixes Tree #2's mistakes, and so on. 
   - **Purpose:** Usually yields the highest accuracy possible for tabular structured CSV data.

---

## 📈 4. Understanding the Evaluation Metrics

When the model is done tuning, it spits out 3 mathematical scores to prove its intelligence:

- **Mean Absolute Error (MAE):** The absolute raw average of how far off our money predictions were. (e.g. "We were off by ₹2,00,000 on average").
- **Root Mean Squared Error (RMSE):** Similar to MAE, but it heavily mathematically penalizes massive outliers. The winning model is usually chosen via the lowest RMSE score!
- **R² Score (Variance):** Represented from 0.0 to 1.0. A score of 0.82 means our model successfully accounts for 82% of all the mathematical reasons why one person makes more money than another in our dataset. 

---

## 🚀 5. How to Run This Project

1. Run the entire pipeline once to train the AI:
   ```bash
   python main.py
   ```
2. Start the interactive User Interface:
   ```bash
   streamlit run app.py
   ```
