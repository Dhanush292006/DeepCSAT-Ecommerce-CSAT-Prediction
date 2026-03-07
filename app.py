import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("csat_model.pkl", "rb"))

st.title("DeepCSAT - Ecommerce Customer Satisfaction Prediction")

st.write("Enter customer support details to predict CSAT Score")

response_time = st.number_input("Response Time")
handling_time = st.number_input("Handling Time")

if st.button("Predict CSAT"):

    features = np.array([[response_time, handling_time]])

    prediction = model.predict(features)

    st.success(f"Predicted CSAT Score: {prediction[0]}")
