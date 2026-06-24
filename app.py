import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load(
    "models/fraud_pipeline.pkl"
)

st.title(
    "Fraud Detection System"
)

transaction_type = st.selectbox(
    "Transaction Type",
    [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]
)

step = st.number_input(
    "Step",
    min_value=1
)

amount = st.number_input(
    "Amount",
    min_value=0.0
)

oldbalanceOrg = st.number_input(
    "Old Balance Origin",
    min_value=0.0
)

newbalanceOrig = st.number_input(
    "New Balance Origin",
    min_value=0.0
)

oldbalanceDest = st.number_input(
    "Old Balance Destination",
    min_value=0.0
)

newbalanceDest = st.number_input(
    "New Balance Destination",
    min_value=0.0
)

newbalanceOrig_missing =  st.number_input(
    "NewBalanceOrigin_missing",
     min_value=0.0
)
isFlaggedFraud = st.number_input(
    "isFlaggedFraud",
    min_value=0
)
oldbalanceDest_missing  = st.number_input(
    "oldbalanceDest_missing",
    min_value=0.0
)

if st.button(
    "Predict Fraud"
):

    data = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "newbalanceOrig_missing": newbalanceOrig_missing,
        "isFlaggedFraud": isFlaggedFraud,
        "oldbalanceDest_missing": oldbalanceDest_missing,
    }])

    prediction = (
        pipeline.predict(data)[0]
    )

    probability = (
        pipeline
        .predict_proba(data)[0][1]
    )

    st.write(
        f"Fraud Probability: {probability:.2%}"
    )

    if prediction == 1:
        st.error(
            "Fraudulent Transaction"
        )
    else:
        st.success(
            "Legitimate Transaction"
        )