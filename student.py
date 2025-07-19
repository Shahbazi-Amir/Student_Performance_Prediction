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
