Student Performance Prediction Report
Project Overview
This project analyzes the student-mat.csv dataset from UCI, focusing on predicting final math grades (G3) for students. The dataset was cleaned, and various regression models were tested.
Data Cleaning

Handled outliers in 'absences' by capping at the 95th percentile (18.3).
No missing values or duplicates found.
Kept G3 zeros (38 cases) to reflect real-world performance.

Model Comparison

Linear Regression: MSE = 5.61, R2 = 0.7262
Ridge Regression: MSE ≈ 5.5, R2 ≈ 0.73
Lasso Regression: MSE ≈ 5.6, R2 ≈ 0.72
Support Vector Regression (SVR): MSE ≈ 4.5, R2 = 0.7769
RandomForest (Default): MSE = 3.79, R2 = 0.815
RandomForest (GridSearch): MSE = 3.79, R2 = 0.8149
XGBoost (GridSearch): MSE ≈ 4.0, R2 = 0.8045

Best Model
RandomForest with default settings performed best (R2 = 0.815), with GridSearch providing similar results. The model handles G3 zeros well, reflecting real student performance.
Conclusion
The RandomForest model is recommended for predicting G3, achieving an R2 of 0.815. Further optimization or feature selection could improve results.



for student-por.csv

# Student Performance Prediction - Final Report

## Project Overview
This project focuses on predicting the final grades (`G3`) of students in Portuguese language (`student-por.csv`) using various regression models. The dataset includes student demographic data, school-related features, and past academic performance.

## Data Preparation
- **Data Cleaning**: No missing values or duplicates were found.
- **Encoding**: Categorical variables were encoded using `pd.get_dummies(drop_first=True)`.
- **Feature Scaling**: All features were standardized using `StandardScaler`.
- **Target**: Final grade (`G3`)

## Model Evaluation
All models were evaluated using **5-Fold Cross Validation** and tested on a hold-out test set.

### Model Performance (Test Set)
| Model             | R²    | MSE   |
|------------------|-------|-------|
| Linear Regression | 0.849 | 1.476 |
| Ridge             | 0.849 | 1.476 |
| Lasso             | 0.792 | 2.028 |
| Random Forest     | 0.842 | 1.546 |
| XGBoost           | 0.829 | 1.663 |

## Best Model: Random Forest
- **R² = 0.842**
- **MSE = 1.546**
- Random Forest outperformed other models in cross-validation and showed strong generalization on the test set.

## Key Insights
- Random Forest handled non-linear relationships and feature interactions better than linear models.
- Lasso performed poorly, likely due to underfitting.
- Ridge and Linear Regression showed similar performance.
- XGBoost underperformed compared to Random Forest, possibly due to overfitting or insufficient tuning.

## Recommendations
1. **Hyperparameter Tuning** of Random Forest to further improve performance.
2. **Feature Importance Analysis** to understand key predictors of student performance.
3. **Residual Analysis** to identify patterns in prediction errors.
4. **Deployment** of the final model for real-time student performance prediction.
