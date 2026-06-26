import pandas as pd
import numpy as np
import os

print("=== Feature Engineering Pipeline for data.csv ===")


file_name = 'data.csv'
if not os.path.exists(file_name):
    print(f"Error: {file_name} not found. Please place it in the same directory.")
    import sys
    sys.exit()

df = pd.read_csv(file_name)
print(f"\n[INFO] Loaded dataset successfully. Shape: {df.shape}")
print("[INFO] Columns available:", list(df.columns))


print("\n--- Step 1: Handling Missing Values ---")
print("Missing values per column before imputation:")
print(df.isnull().sum())

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

median_spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)

median_visits = df['Visits_Per_Month'].median()
df['Visits_Per_Month'] = df['Visits_Per_Month'].fillna(median_visits)

print("Missing values per column after imputation:")
print(df.isnull().sum())

print("\n--- Step 2: Handling Outliers ---")

outlier_mask = df['Age'] > 100
print(f"Number of extreme Age outliers (> 100) identified: {outlier_mask.sum()}")
df.loc[outlier_mask, 'Age'] = median_age

print("\n--- Step 3: Feature Creation ---")


df['Spending_Per_Visit'] = df['Spending'] / df['Visits_Per_Month'].replace(0, 1)


bins = [0, 30, 50, 120]
labels = ['Young', 'Middle-Aged', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

print("New features created: 'Spending_Per_Visit' and 'Age_Group'")
print(df[['Age', 'Age_Group', 'Spending', 'Visits_Per_Month', 'Spending_Per_Visit']].head())


print("\n--- Step 4: Encoding Categorical Variables ---")

categorical_columns = ['City', 'Age_Group']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)


bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print(f"Encoded DataFrame shape: {df_encoded.shape}")


print("\n--- Step 5: Feature Scaling ---")

numerical_columns = ['Age', 'Spending', 'Visits_Per_Month', 'Spending_Per_Visit']

for col in numerical_columns:
    col_mean = df_encoded[col].mean()
    col_std = df_encoded[col].std()
    
    if col_std > 0:
        df_encoded[f'{col}_Scaled'] = (df_encoded[col] - col_mean) / col_std
    else:
        df_encoded[f'{col}_Scaled'] = 0.0

print("Standardized features added (with '_Scaled' suffix):")
print(df_encoded[[f'{col}_Scaled' for col in numerical_columns]].head())

if 'Customer_ID' in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=['Customer_ID'])

output_filename = 'engineered_data.csv'
df_encoded.to_csv(output_filename, index=False)
print(f"\n[SUCCESS] Feature Engineering completed.")
print(f"[SUCCESS] Processed dataset saved to '{output_filename}' with shape {df_encoded.shape}")