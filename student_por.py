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
