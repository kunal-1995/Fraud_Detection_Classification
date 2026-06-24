import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

pipeline = joblib.load(
    "../models/fraud_pipeline.pkl"
)

df = pd.read_csv(
    "../data/test.csv"
)
#Missing Value Imputations
df['type'] = df['type'].fillna('UNKNOWN')
df['nameOrig'] = df['nameOrig'].fillna('MISSING_SENDER')

#Creating new columns as they were used in training dataset
for col in ['newbalanceOrig', 'oldbalanceDest']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

#Applying log transform except for the target column
for col in df.select_dtypes(include = ['number']):
    if col != 'isFraud':
        df[col] = np.log1p(df[col])

drop_cols = [
    "nameOrig",
    "nameDest"
]

existing = [
    c for c in drop_cols
    if c in df.columns
]

df = df.drop(
    columns=existing,
    errors="ignore"
)

X = df.drop(
    "isFraud",
    axis=1
)

y = df["isFraud"]

preds = pipeline.predict(X)

probs = pipeline.predict_proba(X)[:, 1]

print(confusion_matrix(y, preds))

print(classification_report(y, preds))

print(
    "ROC AUC:",
    roc_auc_score(y, probs)
)

print(
    "PR AUC:",
    average_precision_score(y, probs)
)