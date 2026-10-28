import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb
import json
import os


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)


# --------------------------------------------------
# 2. LOAD THE ORIGINAL MACHINE FAILURE MODEL
# --------------------------------------------------

@st.cache_resource
def load_main_model():

    # Load the fitted preprocessing component
    preprocessor = joblib.load("preprocessing.pkl")

    # Load the original XGBoost model
    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")

    return preprocessor, model


# --------------------------------------------------
# 3. LOAD THE FOUR FAILURE-TYPE MODELS
# --------------------------------------------------

@st.cache_resource
def load_failure_models():

    failure_models = {}

    # Read the model paths
    with open(
        "failure_type_models/model_info.json",
        "r"
    ) as file:

        model_info = json.load(file)

    # Load each failure-type model and its preprocessor
    for name, paths in model_info.items():

        failure_preprocessor = joblib.load(
            paths["preprocessor"]
        )

        failure_model = xgb.XGBClassifier()

        failure_model.load_model(
            paths["model"]
        )

        failure_models[name] = {
            "preprocessor": failure_preprocessor,
            "model": failure_model
        }

    return failure_models


# Load the models
try:

    preprocessor, model = load_main_model()
    failure_models = load_failure_models()

except Exception as error:

    st.error("Unable to load the required model files.")

    st.write(
        "Check that all model files are present in the project folder "
        "and that the required Python package versions are installed."
    )

    st.code(str(error))

    st.stop()


# --------------------------------------------------
# 4. APPLICATION HEADING
# --------------------------------------------------

st.title("⚙️ Industrial Machine Failure Prediction")

st.write(
    "Enter the machine's operating conditions to predict "
    "whether it is likely to fail and identify possible failure types."
)

st.divider()


# --------------------------------------------------
# 5. MACHINE INPUT FIELDS
# --------------------------------------------------

st.subheader("Machine Operating Conditions")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"],
    help="L = Low, M = Medium, H = High product quality variants."
)

air_temp = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temp = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rpm = st.number_input(
    "Rotational Speed (rpm)",
    min_value=0,
    max_value=10000,
    value=1500,
    step=10
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=200.0,
    value=40.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=1000,
    value=100,
    step=1
)


# --------------------------------------------------
# 6. FAILURE TYPE NAMES
# --------------------------------------------------

failure_names = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure"
}


# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    # Create a DataFrame from the user's inputs
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    try:

        # ------------------------------------------
        # STAGE 1: PREDICT MACHINE FAILURE
        # ------------------------------------------

        transformed_data = preprocessor.transform(
            input_data
        )

        prediction = model.predict(
            transformed_data
        )[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Prediction: Machine Failure")

            st.subheader("🔍 Possible Failure Types")

            predicted_failures = []

            # --------------------------------------
            # STAGE 2: PREDICT FAILURE INDICATORS
            # --------------------------------------

            for code, components in failure_models.items():

                failure_preprocessor = components[
                    "preprocessor"
                ]

                failure_model = components[
                    "model"
                ]

                # Transform raw input using the
                # corresponding fitted preprocessor
                transformed_input = (
                    failure_preprocessor.transform(
                        input_data
                    )
                )

                # Predict the individual failure indicator
                failure_prediction = failure_model.predict(
                    transformed_input
                )[0]

                if failure_prediction == 1:

                    predicted_failures.append(
                        failure_names.get(code, code)
                    )

            # --------------------------------------
            # DISPLAY FAILURE TYPE PREDICTIONS
            # --------------------------------------

            if predicted_failures:

                for failure in predicted_failures:

                    st.warning(
                        f"⚠️ {failure}"
                    )

            else:

                st.info(
                    "The models did not identify any of the "
                    "four supported failure indicators."
                )

            st.caption(
                "These are experimental model predictions, "
                "not confirmed physical diagnoses. "
                "RNF is excluded because the available training "
                "data was insufficient for reliable prediction."
            )

        else:

            st.success(
                "✅ Prediction: No Machine Failure"
            )

            st.write(
                "The main model predicts no machine failure "
                "under the supplied operating conditions."
            )

    except Exception as error:

        st.error(
            "An error occurred while making the prediction."
        )

        st.code(str(error))
