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

print("Loading production data...")

df = pd.read_csv(
    "../data/production.csv"
)
#Using new columns as they are present while training
for col in ['newbalanceOrig', 'oldbalanceDest']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

#Applying log transform
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
predictions = pipeline.predict(X)

probabilities = (
    pipeline.predict_proba(X)[:, 1]
)

df["prediction"] = predictions

df["fraud_probability"] = probabilities

df.to_csv(
    "../data/production_predictions.csv",
    index=False
)
print(confusion_matrix(y, predictions))

print(classification_report(y, predictions))

print(
    "ROC AUC:",
    roc_auc_score(y, probabilities)
)
print("Scoring complete.")