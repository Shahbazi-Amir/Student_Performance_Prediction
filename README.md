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