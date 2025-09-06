# insurance_premium_prediction

This project provides an end-to-end machine learning solution to predict individual medical insurance costs based on demographic and lifestyle attributes[10].

## Project Overview

- **Goal**: Predict the `expenses` (medical costs) using features like age, sex, BMI, children, smoker status, and region[10].
- **Tech Stack**: Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, seaborn, matplotlib.

## Dataset

The dataset, `insurance_data.csv`, contains 1338 records with these columns:

- **age**: Age of the insured individual
- **sex**: Gender (`male`/`female`)
- **bmi**: Body Mass Index value
- **children**: Number of dependents
- **smoker**: Smoking status (`yes`/`no`)
- **region**: US region (`northeast`, `southeast`, `southwest`, `northwest`)
- **expenses**: Annual insurance charges to be predicted[10].

## Features

- Exploratory data analysis and data summary
- Data preprocessing (scaling numerics, encoding categoricals)
- Regression models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting Regressor
  - XGBoost
  - SVR
- Pipeline with `ColumnTransformer` and `Pipeline`
- Model evaluation: MAE, RMSE, R²

## Workflow

1. **Data Loading**
    - Import `insurance_data.csv` into a pandas DataFrame.
2. **Preprocessing**
    - Standardize numeric variables (`age`, `bmi`, `children`) with `StandardScaler`
    - One-hot encode categorical variables (`sex`, `smoker`, `region`)
    - Use `ColumnTransformer` and `Pipeline` for processing
3. **Modeling & Evaluation**
    - Train and evaluate multiple regression models
    - Use train-test split and cross-validation
4. **Interpretability**
    - Exploratory statistics and visualizations

## Usage

### Prerequisites

Install with pip:

 ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels joblib
    ```


### Running

1. Ensure `insurance_data.csv` is in your working directory.
2. Open `Untitled14.ipynb` in Jupyter Lab or Google Colab.
3. Run notebook cells in order to build models and view results.

## Results

- Provides comparative model metrics (e.g., MAE, RMSE, R²) so you can select the best regression model for insurance cost prediction[10].



For more details, see the notebook for annotated code and plots[10].

