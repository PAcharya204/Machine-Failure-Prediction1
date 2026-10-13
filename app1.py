import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb

# Load the preprocessing component
preprocessor = joblib.load("preprocessing.pkl")

# Load the XGBoost model saved in JSON format
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")

# Configure the page
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

# Application heading
st.title("⚙️ Industrial Machine Failure Prediction")

st.write(
    "Enter the machine's operating conditions to predict "
    "whether it is likely to fail."
)

# Input fields
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temp = st.number_input(
    "Air Temperature (K)",
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature (K)",
    value=310.0
)

rpm = st.number_input(
    "Rotational Speed (rpm)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    value=100
)

# Prediction
if st.button("Predict Machine Failure"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Apply the fitted preprocessing transformations
    transformed_data = preprocessor.transform(input_data)

    # Predict using XGBoost
    prediction = model.predict(transformed_data)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Machine Failure")
    else:
        st.success("✅ Prediction: No Machine Failure")