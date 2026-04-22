import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

model = joblib.load(os.path.join(os.path.dirname(__file__), "random_forest_model.joblib"))

st.title("Churn Prediction App")
st.markdown("---")
st.header("Enter Customer Details:")

credit_score = st.slider("Credit Score", 300, 850, 650)
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", value=0.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
is_active_member = st.radio("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", value=0.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.radio("Gender", ["Male", "Female"])
threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)

has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card_val],
    "IsActiveMember": [is_active_member_val],
    "EstimatedSalary": [estimated_salary]
})

input_data['Geography_Germany'] = np.where(geography == 'Germany', 1, 0)
input_data['Geography_Spain'] = np.where(geography == 'Spain', 1, 0)
input_data['Geography_France'] = np.where(geography == 'France', 1, 0)
input_data['Gender_Male'] = np.where(gender == 'Male', 1, 0)
input_data['Gender_Female'] = np.where(gender == 'Female', 1, 0)

input_data['Balance_to_Salary'] = input_data['Balance'] / (input_data['EstimatedSalary'] + 1e-5)
input_data['CreditScore_to_Age'] = input_data['CreditScore'] / (input_data['Age'] + 1e-5)
input_data['Age_to_Tenure'] = input_data['Age'] / (input_data['Tenure'] + 1e-5)
input_data['CreditScore_Age'] = input_data['CreditScore'] * input_data['Age']
input_data['Balance_Age'] = input_data['Balance'] * input_data['Age']
input_data['Products_Tenure'] = input_data['NumOfProducts'] * input_data['Tenure']
input_data['High_Balance'] = np.where(input_data['Balance'] > 130000, 1, 0)

prediction_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict"):
    probs = model.predict_proba(prediction_data)[:, 1]
    st.info(f"Churn Probability: {probs[0]:.2f}")
    prediction = (probs >= threshold).astype(int)
    if prediction[0] == 1:
        st.error("⚠️ This customer is likely to **leave** the bank.")
    else:
        st.success("✅ This customer is likely to **stay** with the bank.")
