import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from preprocess import get_preprocessor
import numpy as np
print("Loading train sample...")

df = pd.read_csv(
    "../data/train_sample.csv"
)
#missing value imputation
# categorical
df['type'] = df['type'].fillna('UNKNOWN')
df['nameOrig'] = df['nameOrig'].fillna('MISSING_SENDER')

# numerical
for col in ['newbalanceOrig', 'oldbalanceDest']:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

#we cannot remove outliers as removing them means we wont be able to detect big fraudulent transactions
#Hence we will apply log transform as numbers are highly rightly skewed
#IN this dataset as we are not removing outliers, we will go with XGBoost
for col in df.select_dtypes(include = ['number']):
    if col != 'isFraud':
        df[col] = np.log1p(df[col])

# Remove IDs
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

preprocessor = get_preprocessor()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',
    eval_metric='logloss',
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("Training started...")

pipeline.fit(X, y)

joblib.dump(
    pipeline,
    "../models/fraud_pipeline.pkl"
)
print(X.columns.tolist())
print("Model saved.")