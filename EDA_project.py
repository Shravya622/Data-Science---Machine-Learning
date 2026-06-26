import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

print("Understanding dataset")

file_name = 'sales.csv'
if not os.path.exists(file_name):
    print(f"error: {file_name} is not found")
    sys.exit()

df = pd.read_csv(file_name)
print("Successfully loaded ")
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.head())
print(df.tail())
print(df.describe())

print("Handling Missing Values")

print(df.isnull().sum())

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)

mean_spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_spending)
print(mean_spending)

plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10,color='skyblue',edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Spending Amount ')
plt.ylabel('Number of Customers')
plt.show()

correlation = df.corr(numeric_only=True)
print(correlation)

print("Plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title('Correlation Heatmap')
plt.show() 

plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'],color='lightgreen') 
plt.title('Boxplot of customer Age')
plt.xlabel('Age')
plt.show()
