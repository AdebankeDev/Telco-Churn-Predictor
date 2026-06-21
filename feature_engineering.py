"""
Feature engineering for the Telco Customer Churn model.

This mirrors exactly the transformations applied in the training notebook,
so a raw customer record (single form entry or CSV row) can be turned into
the same feature set the model was trained on.
"""

import pandas as pd

INTERNET_COLS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

RAW_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Frozen from the training set (Telco-Customer-Churn.csv, full 7043 rows).
TRAINING_MONTHLY_CHARGES_MEDIAN = 70.35


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --- Type cleaning ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Normalize "No xxx service" labels to "No" (matches training cleanup)
    for col in INTERNET_COLS:
        df[col] = df[col].replace({'No internet service': 'No'})
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})


    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'],
        include_lowest=True
    )

    df['num_services'] = (df[INTERNET_COLS] == 'Yes').sum(axis=1)
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsNewCustomer'] = (df['tenure'] < 12).astype(int)
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    df['high_risk'] = (
        (df['Contract'] == 'Month-to-month') &
        (df['MonthlyCharges'] > TRAINING_MONTHLY_CHARGES_MEDIAN)
    ).astype(int)

    return df
