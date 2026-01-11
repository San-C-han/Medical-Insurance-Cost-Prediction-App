# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:35:44 2026

@author: Priya
"""

import pickle
import streamlit as st
import numpy as np

# Load trained model
model = pickle.load(open(r"C:\Users\Priya\Desktop\ITV\ML Projects\medical_insurance Cost Prediction model\medical_insurance_model.sav","rb"))

# Function to predict insurance cost
def predict_insurance_cost(input_data):
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("💊 Medical Insurance Cost Prediction App")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=100)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])

    # Encoding categorical variables (matching training)
    sex_male = 1 if sex == "male" else 0
    smoker_yes = 1 if smoker == "yes" else 0

    if st.button("Predict Insurance Cost"):
        cost = predict_insurance_cost([age, bmi, children,sex_male, smoker_yes])
        st.success(f"Estimated Medical Insurance Cost: ₹ {cost:,.2f}")

if __name__ == "__main__":
    main()
