# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained flood prediction model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('flood_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load scaler if necessary
@st.cache(allow_output_mutation=True)
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Title of the Streamlit app
st.title('Flood Prediction System')

# Description
st.write("""
This app predicts the likelihood of a flood based on input environmental factors such as rainfall, river level, and other parameters.
""")

# Get user input for the features required by the model
def user_input_features():
    rainfall = st.slider('Rainfall (in mm)', 0, 500, 100)
    river_level = st.slider('River Level (in meters)', 0.0, 10.0, 5.0)
    soil_moisture = st.slider('Soil Moisture (%)', 0, 100, 50)
    humidity = st.slider('Humidity (%)', 0, 100, 75)
    temperature = st.slider('Temperature (°C)', -10, 50, 25)
    
    # Store inputs into a dataframe
    data = {
        'rainfall': rainfall,
        'river_level': river_level,
        'soil_moisture': soil_moisture,
        'humidity': humidity,
        'temperature': temperature
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Store user input features
input_df = user_input_features()

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Display the user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Scaling the input data (if scaler is used in training)
scaled_input = scaler.transform(input_df)

# Predict flood likelihood
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Display the prediction
st.subheader('Flood Prediction Probability')
flood_prob = prediction_proba[0][1] * 100
st.write(f'The probability of a flood is: {flood_prob:.2f}%')

# Final decision based on the model prediction
if flood_prob > 50:
    st.warning("There is a high risk of flooding. Take precautions!")
else:
    st.success("Flood risk is low.")