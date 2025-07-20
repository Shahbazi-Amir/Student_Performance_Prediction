import pandas as pd

df = pd.read_csv('dataset/student-mat.csv')

print("Shape of dataset:", df.shape)

df.head()
# Correctly load the CSV with semicolon separator and quoted strings
df = pd.read_csv('dataset/student-mat.csv', sep=';', quotechar='"')

# Check the shape
print("Shape of dataset:", df.shape)

# Display first 5 rows
df.head()
# Check data types of each column
print("Data types:\n")
print(df.dtypes)

# Check for missing values
print("\nMissing values per column:\n")
print(df.isnull().sum())
# Get statistical summary for numeric columns
df.describe()

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('dataset/student-mat.csv', sep=';', quotechar='"')

# 1. Check for duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

# 2. Check unique values in categorical columns to ensure consistency
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                      'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                      'famsup', 'paid', 'activities', 'nursery', 'higher', 
                      'internet', 'romantic']

print("\nUnique values in categorical columns:")
for col in categorical_columns:
    print(f"{col}: {df[col].unique()}")

# 3. Visualize outliers for key numerical columns
numerical_columns = ['age', 'absences', 'G1', 'G2', 'G3']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# 4. Optional: Handle outliers (example: cap absences at 95th percentile)
absences_95th = df['absences'].quantile(0.95)
df['absences_capped'] = df['absences'].clip(upper=absences_95th)
print(f"95th percentile for absences: {absences_95th}")
print("After capping absences:\n", df[['absences', 'absences_capped']].describe())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset/student-mat.csv', sep=';', quotechar='"')

# Cap absences at 95th percentile
df['absences'] = df['absences'].clip(upper=df['absences'].quantile(0.95))

# Check G3 for outliers
print("G3 statistics:\n", df['G3'].describe())
print("Number of G3 = 0:", (df['G3'] == 0).sum())

# Plot boxplot for G3
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['G3'])
plt.title('Boxplot of G3')
plt.show()


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('dataset/student-mat.csv', sep=';', quotechar='"')

# Cap absences
df['absences'] = df['absences'].clip(upper=df['absences'].quantile(0.95))

# Define categorical and numerical columns
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 
               'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

# Label encode binary columns
label_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encode nominal columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), nominal_cols)
    ], remainder='passthrough')

# Apply one-hot encoding
df_transformed = preprocessor.fit_transform(df)

# Get new column names
onehot_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(nominal_cols)
all_cols = list(onehot_cols) + [col for col in df.columns if col not in nominal_cols]
df = pd.DataFrame(df_transformed, columns=all_cols)

# Standardize numerical columns
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Verify transformed dataset
print("Transformed dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample of transformed data:\n", df.head())



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load transformed dataset (assuming df is the transformed one from previous step)
# If not, re-run the feature engineering code to get df

# Define features and target
X = df.drop('G3', axis=1)  # Features (all columns except G3)
y = df['G3']  # Target (G3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming df is the transformed dataset from previous step
# Define features and target
X = df.drop('G3', axis=1)
y = df['G3']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot 1: Scatter plot of actual vs predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs Predicted G3')
plt.show()

# Plot 2: Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar')
plt.title('Top 10 Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()



