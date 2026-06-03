import streamlit as st
import pandas as pd
import pickle

st.title("Telco Customer Churn Prediction")

model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
st.write("Enter customer details")

gender = st.selectbox("Gender", df["gender"].unique())
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", df["Partner"].unique())
dependents = st.selectbox("Dependents", df["Dependents"].unique())
tenure = st.number_input("Tenure", 0, 72, 2)

contract = st.selectbox("Contract", df["Contract"].unique())
payment = st.selectbox("Payment Method", df["PaymentMethod"].unique())
internet = st.selectbox("Internet Service", df["InternetService"].unique())
tech_support = st.selectbox("Tech Support", df["TechSupport"].unique())
online_security = st.selectbox("Online Security", df["OnlineSecurity"].unique())

monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [df["PhoneService"].mode()[0]],
    "MultipleLines": [df["MultipleLines"].mode()[0]],
    "InternetService": [internet],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [df["OnlineBackup"].mode()[0]],
    "DeviceProtection": [df["DeviceProtection"].mode()[0]],
    "TechSupport": [tech_support],
    "StreamingTV": [df["StreamingTV"].mode()[0]],
    "StreamingMovies": [df["StreamingMovies"].mode()[0]],
    "Contract": [contract],
    "PaperlessBilling": [df["PaperlessBilling"].mode()[0]],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total],
    "tenure_group": ["0-1 Year"],
    "high_value_customer": ["Yes" if monthly > 70 else "No"],
    "total_services": [3]
})

input_data = input_data[feature_columns]

for col in input_data.columns:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Customer will churn. Probability: {probability * 100:.2f}%")

        st.subheader("Suggestions")
        if contract == "Month-to-month":
            st.write("Offer 1-year or 2-year contract discount.")
        if tenure < 12:
            st.write("Improve onboarding and give welcome benefits.")
        if monthly > 70:
            st.write("Provide cheaper plan or monthly bill discount.")
        if payment == "Electronic check":
            st.write("Encourage automatic payment.")
        if tech_support == "No":
            st.write("Offer free tech support trial.")
    else:
        st.success(f"Customer will not churn. Probability: {probability * 100:.2f}%")
        st.write("Maintain service quality and provide loyalty rewards.")
