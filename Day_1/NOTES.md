# Project Study Guide & Code Explanation Notes

This document provides a detailed, step-by-step explanation of the Python code written for both the **Exploratory Data Analysis (EDA)** and the **Feature Engineering Pipeline**. Use this guide to understand and explain the implementation details to your mentor/guide.

---

## 📌 Project Overview
This project processes customer transactional data. We have two datasets:
1. `sales.csv` (used for Exploratory Data Analysis)
2. `data.csv` (used for building the Feature Engineering Pipeline)

Each dataset contains customer profiles:
* **`Customer_ID`**: Numeric ID (arbitrary unique identifier)
* **`Age`**: Age of the customer (contains missing values & outliers)
* **`City`**: Geographical location (categorical text)
* **`Spending`**: Total amount spent (contains missing values)
* **`Visits_Per_Month`**: Frequency of customer visits (contains missing values)

---

## 💡 Key Definitions: What are EDA and Feature Engineering?

### 1. What is Exploratory Data Analysis (EDA)?
**Exploratory Data Analysis (EDA)** is the crucial first step in any data science project. It is the process of examining, cleaning, summarizing, and visualizing a dataset to understand its characteristics before building any predictive models.
* **Why do we do it?** 
  * To understand the "shape" and structure of the data (how many rows, columns, and what types of data we have).
  * To find hidden patterns or relationships (e.g., does spending increase with age?).
  * To detect mistakes, missing values, or weird values (outliers) like an age of 120.
* **Analogy:** EDA is like a detective inspecting a crime scene or a doctor examining a patient before prescribing medication. You need to know the state of the data before trying to fix or model it.

### 2. What is Feature Engineering?
**Feature Engineering** is the process of preparing and transforming raw data features to make them suitable and optimal for Machine Learning algorithms. It involves selecting, modifying, creating, and scaling columns (features).
* **Why do we do it?**
  * Machine learning models are mathematical equations; they cannot directly read strings like "Chicago" or understand categories without numbers.
  * Raw data is often messy, has different scales (e.g., age is 20-80, while spending is in the thousands), or has missing values. Feature engineering standardizes and formats these features so models can learn from them correctly and efficiently.
* **Analogy:** Raw data is like raw ingredients. Machine learning models cannot digest raw data easily. Feature engineering is the process of washing, peeling, chopping, and cooking those ingredients so they are ready to be consumed by the model.

---

## 🔍 Phase 1: Exploratory Data Analysis (EDA)
**File Location:** [EDA_project.py](file:///c:/DS-ML/Data-Science---Machine-Learning/EDA_project.py)

The goal of EDA is to load the dataset, understand its shape, check for missing values, handle basic data issues, and plot visualizations to discover patterns and anomalies.

### 💻 Step-by-Step Code Explanation

#### 1. Importing Libraries & Checking File Existence
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

file_name = 'sales.csv'
if not os.path.exists(file_name):
    print(f"error: {file_name} is not found")
    sys.exit()
```
* **Explanation:**
  * `pandas` is imported for data structures (DataFrames) and file reading.
  * `matplotlib.pyplot` and `seaborn` are standard visualization libraries.
  * `os.path.exists` checks if the file exists in the directory. If it doesn't, we stop execution (`sys.exit()`) to avoid a crash.

---

#### 2. Loading and Inspecting the Data
```python
df = pd.read_csv(file_name)
print("Successfully loaded ")
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.head())
print(df.tail())
print(df.describe())
```
* **Explanation:**
  * `pd.read_csv()` reads the CSV file into a DataFrame `df`.
  * `df.shape` returns a tuple `(rows, columns)`. `shape[0]` represents the row count, and `shape[1]` represents the column count.
  * `df.head()` displays the first 5 rows; `df.tail()` displays the last 5 rows.
  * `df.describe()` generates descriptive summary statistics (count, mean, standard deviation, min, 25%, 50% (median), 75%, and max) for numeric columns.

---

#### 3. Finding and Handling Missing Values
```python
print(df.isnull().sum())

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

mean_spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_spending)
```
* **Explanation:**
  * `df.isnull().sum()` checks each column for missing (`NaN`) values and returns the count of missing items in each column.
  * **Age Imputation:** We compute the median age using `.median()` and fill missing values with it using `.fillna()`. We use the **median** because it represents the middle value and is not affected by outliers.
  * **Spending Imputation:** We compute the average spending using `.mean()` and fill missing values with it.

---

#### 4. Visualizing Data Distributions and Relationships
```python
# Histogram of Spending
plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Spending Amount ')
plt.ylabel('Number of Customers')
plt.show()
```
* **Explanation:**
  * `plt.figure(figsize=(7,4))` sets the dimensions of the chart window.
  * `df['Spending'].hist(bins=10)` splits the range of spending values into 10 intervals (bins) and plots bars representing the number of customers in each bin.

```python
# Correlation Heatmap
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(7,4))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
```
* **Explanation:**
  * `df.corr(numeric_only=True)` calculates the Pearson correlation coefficient matrix between all numeric columns.
  * `sns.heatmap()` displays this matrix visually. `annot=True` writes the numerical values in each cell, and `cmap='coolwarm'` sets the color scale (blue for negative correlation, red for positive correlation).

```python
# Boxplot for Outlier Detection
plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'], color='lightgreen') 
plt.title('Boxplot of customer Age')
plt.xlabel('Age')
plt.show()
```
* **Explanation:**
  * A **boxplot** represents data quartiles. The box shows the Interquartile Range (IQR, 25th to 75th percentile), the middle line is the median, and the whiskers show the typical range.
  * *Finding:* Points outside the whiskers indicate extreme values. In this plot, we found an extreme outlier in `Age` (greater than 100).

---

## ⚙️ Phase 2: Feature Engineering Pipeline
**File Location:** [Feature_Engineering.py](file:///c:/DS-ML/Data-Science---Machine-Learning/Feature_Engineering.py)

The goal of feature engineering is to clean the dataset, transform values, create new descriptive features, and scale the data so that it can be fed into machine learning algorithms.

### 💻 Step-by-Step Code Explanation

#### 1. Imputing Missing Values
```python
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

median_spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)

median_visits = df['Visits_Per_Month'].median()
df['Visits_Per_Month'] = df['Visits_Per_Month'].fillna(median_visits)
```
* **Explanation:**
  * Unlike the EDA script, in the pipeline we impute all missing values (`Age`, `Spending`, and `Visits_Per_Month`) using their respective **medians** to ensure outliers do not skew the imputed values.

---

#### 2. Capping Age Outliers
```python
outlier_mask = df['Age'] > 100
df.loc[outlier_mask, 'Age'] = median_age
```
* **Explanation:**
  * We create a boolean mask (`outlier_mask`) where `True` indicates the customer's age is greater than 100.
  * `df.loc[outlier_mask, 'Age'] = median_age` locates the rows where the mask is `True`, and replaces their age with the `median_age` (calculated in Step 1). This cleans the dataset of unrealistic age values (e.g., 110, 120).

---

#### 3. Creating New Features (Feature Generation)
```python
df['Spending_Per_Visit'] = df['Spending'] / df['Visits_Per_Month'].replace(0, 1)

bins = [0, 30, 50, 120]
labels = ['Young', 'Middle-Aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```
* **Explanation:**
  * **`Spending_Per_Visit`**: Calculates how much a customer spends on average per visit.
    * *Coding Trick:* We cannot divide by 0. So we use `df['Visits_Per_Month'].replace(0, 1)` to replace any instance of `0` visits with `1` during division, avoiding a `DivisionByZero` mathematical error.
  * **`Age_Group`**: Discretizes continuous age values into categorical labels.
    * `pd.cut()` takes the `Age` column and maps each row into a bin based on the thresholds: `0-30` (Young), `30-50` (Middle-Aged), and `50-120` (Senior).

---

#### 4. Encoding Categorical Variables
```python
categorical_columns = ['City', 'Age_Group']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
```
* **Explanation:**
  * Machine learning models require numeric matrices. We cannot pass string categories like `'New York'` or `'Young'`.
  * `pd.get_dummies()` performs **One-Hot Encoding**. It creates a binary column for every category (e.g., `City_New York`, `City_Chicago`, `Age_Group_Young`, etc.). If a row belongs to that category, it gets a `True` value; otherwise `False`.
  * Since pandas outputs these categories as boolean data types, we detect all boolean columns using `df_encoded.select_dtypes(include=['bool']).columns` and cast them to integers (`1` or `0`) using `.astype(int)`.

---

#### 5. Numerical Feature Scaling (Standardization)
```python
numerical_columns = ['Age', 'Spending', 'Visits_Per_Month', 'Spending_Per_Visit']

for col in numerical_columns:
    col_mean = df_encoded[col].mean()
    col_std = df_encoded[col].std()
    
    if col_std > 0:
        df_encoded[f'{col}_Scaled'] = (df_encoded[col] - col_mean) / col_std
    else:
        df_encoded[f'{col}_Scaled'] = 0.0
```
* **Explanation:**
  * Features have different scales: age values are low (20-80), but spending values are high (100-8000). ML models might misinterpret the higher-valued features as more important.
  * **Standardization (Z-score normalization)** transforms data to have a mean ($\mu$) of `0` and a standard deviation ($\sigma$) of `1`.
  * **Formula:** $$z = \frac{x - \mu}{\sigma}$$
  * For each numeric column, we calculate its mean and standard deviation.
  * If the standard deviation is 0 (all values are identical), we assign 0.0 to prevent division-by-zero errors. Otherwise, we calculate the Z-score and save it under a new column with the `_Scaled` suffix.

---

#### 6. Drop IDs and Save Data
```python
if 'Customer_ID' in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=['Customer_ID'])

output_filename = 'engineered_data.csv'
df_encoded.to_csv(output_filename, index=False)
```
* **Explanation:**
  * **`Customer_ID`** is unique for every row, acting only as a row identifier. Since it contains no generalizable patterns, keeping it in model training could lead to overfitting. We drop it.
  * `df_encoded.to_csv(..., index=False)` saves the final cleaned and engineered dataset to `engineered_data.csv` without writing row index numbers.

---

## 📝 Critical Concepts to Share with your Guide

1. **Mean Imputation vs. Median Imputation:**
   * Mean imputation works well for normal/symmetrical distributions.
   * Median imputation is preferred for skewed columns (like `Spending`) and columns with outliers (like `Age`), because the median is insensitive to extreme values.

2. **Handling Outliers:**
   * In `EDA_project.py`, we identified an age of `120` via boxplots.
   * In `Feature_Engineering.py`, we solved this by **capping** values over `100` to the dataset's median age. This prevents regression models from shifting their boundaries to fit unreasonable outlier points.

3. **Avoiding Division-By-Zero:**
   * In python, dividing by zero crashes the execution. In `df['Spending'] / df['Visits_Per_Month']`, we handled this cleanly using `.replace(0, 1)` on the denominator.

4. **One-Hot Encoding vs. Ordinal Encoding:**
   * We used One-Hot encoding because categories like cities (`Chicago`, `New York`) or age groups do not have a natural ranking (e.g. Chicago is not "greater than" New York). Ordinal scaling would introduce a false mathematical hierarchy.

5. **Feature Scaling (Standardization):**
   * Normalizes numeric inputs so that gradient descent converges faster and distance calculations (used in algorithms like K-Nearest Neighbors or Support Vector Machines) are not biased toward columns with larger numerical scales.
