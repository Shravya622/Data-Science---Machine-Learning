import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# category_encoders is needed for Target Encoding
try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will be skipped.")

def main():
    print("Loading Dataset...")
    file_path = "train.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    print(
        f"Dataset loaded successfully.\n"
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}"
    )
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Handling Missing Values demonstration
    print("\nHandling Missing DATA...")
    print("Artificially setting some 'Hits' (H) values to NaN for demonstration...")

    # Artificially create missing values
    df.loc[0:24, 'H'] = np.nan

    # Create an imputer that replaces missing values with the median of H
    imputer = SimpleImputer(strategy='median')
    df[['H']] = imputer.fit_transform(df[['H']])
    print(
        f"Imputation complete. "
        f"'H' column now has {df['H'].isnull().sum()} missing values."
    )
    print("\nFirst 30 rows of H:")
    print(df['H'].head(30))

    # high cardinality 
    # creating high cardinality manually
    df['Team_ID'] = [f"Team_{np.random.randint(1, 150)}" for _ in range(len(df))]

    if TargetEncoder is not None:
        print("\nApplying Target Encoder...")

        encoder = TargetEncoder()

        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'], df['W'])
        print("Target encoding complete. First 5 rows:")
        print(df[['Team_ID', 'W', 'Team_ID_Encoded']].head())

    else:
        print("\ncategory Encoder not installed")

    # Feature selection
    features_to_test = ['H', 'HR', 'SO', 'SB']
    x_features = df[features_to_test].fillna(0)
    y_target = df['W']
    
    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(x_features, y_target)
    
    winning_features = selector.get_support()
    best_features = x_features.columns[winning_features].tolist()
    
    print("\nSelected features:")
    print(best_features)

    print(best_features)

    #splitting data
    x=df[best_features]
    y=df['W']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    print(f"training data size:{x_train.shape}")
    print(f"testing data size:{x_test.shape}\n")

if __name__ == '__main__':
    main()
    
    #training model
