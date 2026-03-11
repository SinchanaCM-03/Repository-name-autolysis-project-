import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Get file name from command argument
file_path = sys.argv[1]

# Load dataset
df = pd.read_csv(file_path, encoding="latin1")

print("CSV loaded successfully")

# Print rows and columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics (numeric columns):")
print(df.describe())

# Select numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

# Generate histograms
for col in numeric_cols:
    plt.figure()
    df[col].hist()
    plt.title(col)
    plt.savefig(f"{col}_hist.png")
    plt.close()

print("Histograms saved successfully")

# Correlation matrix
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig("correlation_matrix.png")
plt.close()

print("Correlation matrix saved successfully")

# Outlier Detection using Boxplots
numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title(f"Outlier Detection: {col}")
    plt.savefig(f"{col}_boxplot.png")
    plt.close()

print("Outlier detection boxplots saved successfully")
