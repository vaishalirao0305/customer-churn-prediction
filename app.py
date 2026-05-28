import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

# Inputs
tenure = st.number_input("Tenure", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Prediction

if st.button("Predict"):
    st.write("Button clicked")

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("Prediction done")

    if prediction == 1:
        st.error(f"Customer likely to churn. Probability: {probability:.2f}")
    else:
        st.success(f"Customer likely to stay. Probability: {probability:.2f}")