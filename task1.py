import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("titanic.csv")

# First 5 rows
print("First 5 Rows")
print(df.head())

# Dataset information
print("\nDataset Info")
df.info()

# Missing values
print("\nMissing Values")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary")
print(df.describe())

# -------------------------
# Age Distribution Plot
# -------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Age"].dropna(), kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.savefig("outputs/age_distribution.png")

print("\nAge distribution graph saved successfully!")

plt.show()