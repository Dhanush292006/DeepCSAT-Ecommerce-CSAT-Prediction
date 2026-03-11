import streamlit as st
import numpy as np
import joblib

model = joblib.load("csat_model.pkl")

st.title("DeepCSAT - Customer Satisfaction Prediction")

response_time = st.number_input("Response Time",min_value=0.0)
handling_time = st.number_input("Handling Time",min_value=0.0)

if st.button("Predict CSAT"):

    features = np.array([[response_time,handling_time]])

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")
