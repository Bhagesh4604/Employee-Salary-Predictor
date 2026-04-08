# Employee Salary Prediction Using Machine Learning
**Context:** Indian Job Market (INR)

## 1. Project Description
The goal of this project is to build an intelligent forecasting engine capable of predicting employee salaries based on demographic and professional attributes. The model leverages the well-known UCI Adult Census dataset, which has been mathematically transformed to reflect the Indian corporate salary landscape (ranging from ₹4L to ₹25L+ LPA). By taking in factors such as Age, Education Level, City Tier, Job Role, and Years of Experience, the system provides accurate, data-driven salary estimations. 

## 2. Project Steps
1. **Data Ingestion & Transformation:** Loaded raw census data and applied synthetic transformations to generate a localized target variable (`Salary_INR`).
2. **Exploratory Data Analysis (EDA):** Generated visual insights regarding how education, experience, and city tier correlate with higher compensations.
3. **Data Preprocessing:** Handled missing values, scaled numerical features using Standard Scaler, and applied One-Hot Encoding to categorical variables.
4. **Machine Learning Training:** Evaluated multiple models including Linear Regression, Random Forest, and Gradient Boosting to find the optimal mathematical fit.
5. **Hyperparameter Tuning:** Fine-tuned the best-performing model (Random Forest) via RandomizedSearchCV to maximize predictive accuracy.
6. **Web Application Deployment:** Built a high-end, production-ready Streamlit dashboard featuring custom 3D glassmorphism CSS, minimal inputs, and modern typography.

## 3. Implementation
The project pipeline is modularized into dedicated Python scripts:
* `data_loader.py`: Handles data fetching and localized INR transformation.
* `preprocessing.py`: Manages the scikit-learn transformers and data cleaning pipelines.
* `train.py` & `tuner.py`: Defines the core ML algorithms and executes grid-search optimization.
* `evaluate.py`: Generates the core statistical metrics (MAE, RMSE, R²).
* `inference.py`: Loads the serialized `.pkl` models for real-time predictions.
* `app.py`: The user-facing Streamlit dashboard, styled to mirror a premium Awwwards web template.

## 4. Results
The Machine Learning models achieved strong mathematical convergence during testing:
* **Best Model:** Tuned Random Forest Regressor
* **Mean Absolute Error (MAE):** Represents the average prediction error, minimized significantly post-tuning.
* **Accuracy (R² Score):** Captured a strong variance in the dataset, proving that Education, Job Role, and Experience are the primary drivers of salary variance in India.
* **UI Delivery:** Successfully integrated the trained model into a zero-latency web interface that predicts salaries instantly upon user input.

## 5. Conclusion
This project successfully bridges the gap between raw data science and polished software engineering. By adapting a standard dataset to the Indian context and deploying it through a heavily customized, premium Streamlit dashboard, the resulting application serves as both an accurate analytical tool and a highly engaging user experience. The modular architecture ensures that future data inputs or alternative models can be integrated seamlessly.
