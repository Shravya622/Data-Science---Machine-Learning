# Data Science & Machine Learning Pipeline Project

Welcome to the **Data Science & Machine Learning Pipeline** repository! This project implements a complete, step-by-step pipeline starting with Exploratory Data Analysis (EDA) and moving into a comprehensive Feature Engineering Pipeline for cleaning and transforming customer demographic and transactional data.

---

## 📁 Repository Structure

* 📊 **Data Files:**
  * `sales.csv` - The raw dataset used for Exploratory Data Analysis (EDA).
  * `data.csv` - The raw dataset used as the input for the Feature Engineering Pipeline.
  * `engineered_data.csv` - The final, fully processed, encoded, and scaled output dataset ready for Machine Learning models.
* 🐍 **Python Scripts:**
  * [EDA_project.py](file:///c:/DS-ML/Data-Science---Machine-Learning/EDA_project.py) - Script that conducts data exploration, basic missing value imputation, and plots distribution/correlation/outlier visualizations.
  * [Feature_Engineering.py](file:///c:/DS-ML/Data-Science---Machine-Learning/Feature_Engineering.py) - Script containing the production-ready feature engineering pipeline (Imputation, Outlier Capping, Feature Creation, Categorical Encoding, and Feature Scaling).
* 📝 **Study & Reference Notes:**
  * [NOTES.md](file:///c:/DS-ML/Data-Science---Machine-Learning/NOTES.md) - A detailed, line-by-line guide explaining the code implementation, logic, and core machine learning pre-processing concepts.

---

## ⚙️ Project Pipeline Overview

### 1. Exploratory Data Analysis (EDA)
In the EDA stage, we load `sales.csv` and inspect it to understand data shapes, types, and summary statistics. We handle basic missing values and generate visualizations to guide our pipeline decisions:
* **Distribution of Spending:** A histogram visualizing customer spending ranges.
* **Correlation Heatmap:** A correlation matrix showing standard relationships between numerical columns.
* **Boxplot of Age:** A boxplot used to detect extreme age outliers.

### 2. Feature Engineering Pipeline
Using insights from EDA, the production pipeline processes `data.csv` and outputs `engineered_data.csv` through the following sequence:

1. **Missing Value Imputation:** Imputes missing numerical cells using column-wise **medians**.
2. **Outlier Capping:** Identifies unrealistic age records (Age > 100) and caps them to the median age.
3. **Feature Creation:**
   * `Spending_Per_Visit` = `Spending` / `Visits_Per_Month` (with division-by-zero protection).
   * `Age_Group` = Binned age groups (`Young`, `Middle-Aged`, `Senior`).
4. **Categorical Encoding:** One-hot encodes `City` and `Age_Group` columns, casting boolean values to `1`/`0` integers.
5. **Feature Scaling:** Normalizes numerical features (`Age`, `Spending`, `Visits_Per_Month`, `Spending_Per_Visit`) using Z-score standardization:
   $$z = \frac{x - \mu}{\sigma}$$
6. **Feature Exclusion:** Excludes `Customer_ID` as it is not predictive, and exports the final dataset.

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python installed, along with the required libraries:
```bash
pip install pandas numpy matplotlib seaborn
```

### 🏃 How to Run the Files

1. **To run the Exploratory Data Analysis (EDA) script:**
   ```bash
   python EDA_project.py
   ```
   * *This will output statistics to the console and display matplotlib charts for spending distribution, correlations, and age boxplots.*

2. **To run the Feature Engineering Pipeline:**
   ```bash
   python Feature_Engineering.py
   ```
   * *This will process `data.csv` and generate the final preprocessed output dataset saved as `engineered_data.csv`.*

---

## 📚 Study Notes
For a detailed code walkthrough and an explanation of key machine learning concepts (such as Mean vs. Median Imputation, One-Hot Encoding, and Standardization), please refer to the project study guide: [NOTES.md](file:///c:/DS-ML/Data-Science---Machine-Learning/NOTES.md).