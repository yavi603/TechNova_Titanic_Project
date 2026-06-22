import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/bar_chart.png")

df = pd.read_csv("titanic.csv")

# 1 Bar Chart
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class")
plt.savefig("outputs/bar_chart.png")
plt.show()

# 2 Line Chart
df['Age'].sort_values().reset_index(drop=True).plot()
plt.title("Age Trend")
plt.savefig("outputs/line_chart.png")
plt.show()

# 3 Pie Chart
df['Sex'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.title("Gender Distribution")
plt.savefig("outputs/gender_pie.png")
plt.show()

# 4 Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df['Age'], df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.savefig("outputs/scatter_plot.png")
plt.show()

# 5 Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.show()