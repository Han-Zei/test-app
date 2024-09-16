import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Title and description
st.title('Barangay Accident Prediction Model')
st.write("""
A web app to predict accident-prone barangays based on input factors.
""")

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('res_santiago.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Check for missing values and fill them with 0
    data['Resident'].fillna(0, inplace=True)
    data['Non-Resident'].fillna(0, inplace=True)

    # Create 'Total_Accidents' column
    data['Total_Accidents'] = data['Resident'] + data['Non-Resident']

    return data

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Ensure that the 'Date' column is in the correct format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Check for missing values and fill them with 0
    data['Resident'].fillna(0, inplace=True)
    data['Non-Resident'].fillna(0, inplace=True)
    
    # Create 'Total_Accidents' column
    data['Total_Accidents'] = data['Resident'] + data['Non-Resident']
    
    st.write(data.head())

data = load_data()
st.write("### Dataset Preview")
st.dataframe(data.head())

# Model choice
model_choice = st.sidebar.selectbox("Select Model", ["SARIMAX", "Random Forest", "LSTM"])

# Input Parameters
st.sidebar.header('Input Parameters')
def user_input_features():
    provincial_road = st.sidebar.slider('Provincial Road', 0, 100, 50)
    city_road = st.sidebar.slider('City Road', 0, 100, 50)
    motorcycle = st.sidebar.slider('Motorcycle', 0, 100, 50)
    return {'Provincial Road': provincial_road, 'City Road': city_road, 'Motorcycle': motorcycle}

inputs = user_input_features()
st.write("### User Inputs")
st.write(inputs)

# Prediction logic for each model
def predict_accident_prone_barangays(inputs, model_choice):
    prone_barangays = []
    
    if model_choice == "Random Forest":
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        prone_barangays = ['Barangay 1', 'Barangay 2', 'Barangay 5']  # Example prone barangays
        
    elif model_choice == "SARIMAX":
        # Assuming SARIMAX model has been trained on time-series data
        sarimax_model = SARIMAX(data['Total_Accidents'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarimax_model_fit = sarimax_model.fit(disp=False)
        prone_barangays = ['Barangay 3', 'Barangay 4']  # Example prone barangays for SARIMAX

    elif model_choice == "LSTM":
        # Assuming LSTM model has been trained on relevant data
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 3)))
        lstm_model.add(Dense(1))
        prone_barangays = ['Barangay 6', 'Barangay 7']  # Example prone barangays for LSTM

    return prone_barangays

# Display accident-prone barangays
prone_barangays = predict_accident_prone_barangays(inputs, model_choice)
st.write(f"### Predicted Accident-Prone Barangays: {', '.join(prone_barangays)}")

# Plotting the data
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Total_Accidents'], label='Total Accidents')
ax.set_xlabel('Date')
ax.set_ylabel('Total Accidents')
ax.set_title('Accidents Over Time')
ax.legend()

st.pyplot(fig)
