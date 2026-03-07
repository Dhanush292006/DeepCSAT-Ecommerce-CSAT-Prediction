import streamlit as st
import numpy as np
import joblib

st.title("DeepCSAT - Customer Satisfaction Prediction")

try:
    model = joblib.load("csat_model.pkl")
except:
    st.error("Model file not found. Please upload csat_model.pkl to the repository.")
    st.stop()

response_time = st.number_input("Response Time")
handling_time = st.number_input("Handling Time")

if st.button("Predict CSAT"):

    features = np.array([[response_time, handling_time]])

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")
