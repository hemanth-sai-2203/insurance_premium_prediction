import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("GradientBoosting_Best_Model.pkl")

st.title("💰 Insurance Expense Prediction App")
st.write("Predict medical insurance expenses based on user inputs.")

# Sidebar Inputs
st.sidebar.header("Input Features")

age = st.sidebar.slider("Age", 18, 65, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare input data as a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

st.write("### Input Summary")
st.dataframe(input_data)

# Prediction
if st.button("Predict Expenses"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Expense: **${prediction[0]:,.2f}**")
