# Import pandas
import pandas as pd

# Load the Portuguese language dataset
df_por = pd.read_csv('dataset/student-por.csv', sep=';', quotechar='"')

# Display shape and first rows
print("Shape of Portuguese dataset:", df_por.shape)
df_por.head()
# Check data types of all columns
df_por.dtypes
# Check for missing values in each column
df_por.isnull().sum()
# Check for duplicate rows
df_por.duplicated().sum()
# Summary statistics for numerical features
df_por.describe()
# Visualize the distribution of final grades (G3) in Portuguese subject
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(df_por['G3'], bins=20, kde=True)
plt.title("Distribution of Portuguese Final Grade (G3)")
plt.xlabel("G3")
plt.ylabel("Count")
plt.grid(True)
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset/student-por.csv', sep=';')

# Select numeric columns
numeric_cols = ['age', 'absences', 'G1', 'G2', 'G3', 'Dalc', 'Walc']

# Plot boxplots for outlier detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)
    sns.boxplot(data=df, y=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()
# Histogram of absences
plt.figure(figsize=(8, 4))
sns.histplot(df['absences'], bins=30, kde=True)
plt.title('Distribution of Absences')
plt.xlabel('Absences')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Relationship between G1, G2 and G3
sns.pairplot(df, vars=['G1', 'G2', 'G3'], kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.suptitle('Relationship between G1, G2, and G3', y=1.02)
plt.show()
df_por_cleaned = df_por.copy()


df_por_cleaned = df_por.copy()


df_por_encoded = pd.get_dummies(df_por_cleaned, drop_first=True)


X = df_por_encoded.drop(['G3'], axis=1)
y = df_por_encoded['G3']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
print("Linear Regression R2:", r2_score(y_test, y_pred))



from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("Ridge Regression MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Ridge Regression R2:", r2_score(y_test, y_pred_ridge))



from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("Lasso Regression MSE:", mean_squared_error(y_test, y_pred_lasso))
print("Lasso Regression R2:", r2_score(y_test, y_pred_lasso))


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))



from xgboost import XGBRegressor

xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost'],
    'MSE': [
        mean_squared_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred_ridge),
        mean_squared_error(y_test, y_pred_lasso),
        mean_squared_error(y_test, y_pred_rf),
        mean_squared_error(y_test, y_pred_xgb)
    ],
    'R2': [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_ridge),
        r2_score(y_test, y_pred_lasso),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb)
    ]
})
print(results.sort_values(by='R2', ascending=False))


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline با استانداردسازی
pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso())
])

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),  # اختیاری برای Random Forest
    ('rf', RandomForestRegressor())
])

# فیت کردن
pipeline_lasso.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)

# امتیازدهی
print("Lasso with scaling:", pipeline_lasso.score(X_test, y_test))
print("Random Forest with scaling:", pipeline_rf.score(X_test, y_test))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer

# 1. بارگذاری داده
df = pd.read_csv('dataset/student-por.csv', sep=';')

# 2. کدگذاری متغیرهای کتگوریکی
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. جدا کردن ویژگی‌ها و هدف
X = df_encoded.drop('G3', axis=1)
y = df_encoded['G3']

# 4. استانداردسازی داده
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. تعریف مدل‌ها
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# 6. تعریف معیارهای ارزیابی
scoring = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error)
}

# 7. ارزیابی مدل‌ها با Cross Validation
results = {}

for name, model in models.items():
    cv_results = cross_validate(model, X_scaled, y, cv=5, scoring=scoring)
    r2_scores = cv_results['test_r2']
    mse_scores = cv_results['test_mse']
    results[name] = {
        'Mean R2': np.mean(r2_scores),
        'Mean MSE': np.mean(mse_scores)
    }
    print(f"{name}: R2={np.mean(r2_scores):.4f}, MSE={np.mean(mse_scores):.4f}")
    
    
    import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. بارگذاری داده
df_por = pd.read_csv('dataset/student-por.csv', sep=';')

# 2. کدگذاری متغیرهای کتگوریکی
df_encoded = pd.get_dummies(df_por, drop_first=True)

# 3. جدا کردن ویژگی‌ها (X) و هدف (y)
X = df_encoded.drop('G3', axis=1)
y = df_encoded['G3']

# 4. تقسیم داده به train و test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. استانداردسازی داده
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

# تعریف مدل‌ها
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# معیارهای ارزیابی
scoring = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error)
}

# اجرای کراس ولیدیشن
results = {}

for name, model in models.items():
    cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, scoring=scoring)
    r2_scores = cv_results['test_r2']
    mse_scores = cv_results['test_mse']
    results[name] = {
        'Mean R2': np.mean(r2_scores),
        'Mean MSE': np.mean(mse_scores)
    }
    print(f"{name}: R2={np.mean(r2_scores):.4f}, MSE={np.mean(mse_scores):.4f}")
    
    
    
# آموزش و تست مدل‌ها
test_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    test_results[name] = {'Test R2': r2, 'Test MSE': mse}
    print(f"{name} (Test): R2={r2:.4f}, MSE={mse:.4f}")
    
    
    
results_df = pd.DataFrame(results).T
test_df = pd.DataFrame(test_results).T

# ترکیب نتایج
final_results = results_df.join(test_df, lsuffix='_CV', rsuffix='_Test')
print(final_results)