import pandas as pd
import joblib

pipeline = joblib.load(
    "../models/fraud_pipeline.pkl"
)


def predict_transaction(data):

    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)[0]

    probability = (
        pipeline
        .predict_proba(df)[0][1]
    )

    return prediction, probability


if __name__ == "__main__":

    sample = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 10000,
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 10000,
        "newbalanceOrig_missing": 0 ,
        "isFlaggedFraud": 1,
        "oldbalanceDest_missing": 0,
    }

    pred, prob = predict_transaction(sample)

    print(pred)
    print(prob)