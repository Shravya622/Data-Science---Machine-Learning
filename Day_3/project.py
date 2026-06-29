import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# category_encoders is needed for Target Encoding
try:
    from category_encoders import TargetEncoder  # type: ignore[import-untyped]  # pyrefly: ignore[missing-import]
    _TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TargetEncoder = None  # type: ignore[assignment,misc]
    _TARGET_ENCODER_AVAILABLE = False
    print("Warning: category_encoders not installed. Target Encoding will be skipped.")


def main() -> None:
    print("Loading Dataset...")

    # Load California Housing from sklearn (includes target: MedHouseVal)
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # full DataFrame with features + target column

    print(
        f"Dataset loaded successfully.\n"
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}"
    )
    print("\nColumn names:")
    print(df.columns.tolist())

    # ─── HANDLING MISSING VALUES ───────────────────────────────────────────────
    print("\nHandling Missing Data...")
    print("Artificially setting some 'AveBedrms' values to NaN for demonstration...")

    # Artificially create missing values in a real column
    df.loc[0:24, 'AveBedrms'] = np.nan

    # Impute missing values with the median of that column
    imputer = SimpleImputer(strategy='median')
    df[['AveBedrms']] = imputer.fit_transform(df[['AveBedrms']])
    print(
        f"Imputation complete. "
        f"'AveBedrms' now has {df['AveBedrms'].isnull().sum()} missing values."
    )
    print("\nFirst 5 rows of AveBedrms after imputation:")
    print(df['AveBedrms'].head(5))

    # ─── TARGET ENCODING (High Cardinality Demo) ──────────────────────────────
    # Create a fake high-cardinality categorical column to demonstrate Target Encoding
    df['Region_ID'] = [f"Region_{np.random.randint(1, 50)}" for _ in range(len(df))]

    if _TARGET_ENCODER_AVAILABLE and TargetEncoder is not None:
        print("\nApplying Target Encoding...")
        encoder = TargetEncoder()
        df['Region_ID_Encoded'] = encoder.fit_transform(  # type: ignore[union-attr]
            df['Region_ID'], df['MedHouseVal']
        )
        print("Target encoding complete. First 5 rows:")
        print(df[['Region_ID', 'MedHouseVal', 'Region_ID_Encoded']].head())
    else:
        print("\nCategory Encoders not installed. Skipping Target Encoding.")

    # ─── FEATURE SELECTION ────────────────────────────────────────────────────
    features_to_test = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                        'Population', 'AveOccup']
    target_column = 'MedHouseVal'  # median house value (in $100,000s)

    x_features = df[features_to_test].fillna(0)
    y_target = df[target_column]

    print("\nRunning Feature Selection (SelectKBest with mutual_info_regression)...")
    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(x_features, y_target)

    winning_features = selector.get_support()
    best_features = x_features.columns[winning_features].tolist()

    print(f"Top 2 selected features: {best_features}")

    # ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────
    X = df[best_features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining data size : {X_train.shape}")
    print(f"Testing  data size : {X_test.shape}")

    # ─── TRAIN LINEAR REGRESSION MODEL ────────────────────────────────────────
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Show first 3 predictions vs actual values
    print("\n--- Sample Predictions (house value in $100,000s) ---")
    actual_values = y_test.head(3).values
    predicted_values = predictions[:3]

    for i in range(3):
        predicted = round(predicted_values[i], 2)
        actual = round(actual_values[i], 2)
        difference = round(abs(actual - predicted), 2)

        print(f"  Model Predicted : ${predicted * 100_000:,.0f}")
        print(f"  Actual Value    : ${actual   * 100_000:,.0f}")
        print(f"  Difference      : ${difference * 100_000:,.0f}")
        print()


if __name__ == '__main__':
    main()