import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


# -----------------------------
# Load saved model and artifacts
# -----------------------------
model = joblib.load(MODEL_DIR / "best_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
expected_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")



numeric_features = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Heart Disease Prediction")

st.write(
    "Enter the details below to get an educational "
    "prediction from the trained machine learning model."
)

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=0,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=0,
    max_value=700,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider("Maximum Heart Rate", 50, 220, 150)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    # Create the numeric input features
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak
    }

    # Match pandas get_dummies(drop_first=True):
    # create dummy columns only for categories represented
    # in the training feature list.
    category_values = {
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }

    for feature, value in category_values.items():
        dummy_column = f"{feature}_{value}"

        if dummy_column in expected_columns:
            input_data[dummy_column] = 1

    # Create input DataFrame
    input_df = pd.DataFrame([input_data])

    # Add any missing training columns as zero
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep exactly the training column order
    input_df = input_df[expected_columns]

    # Scale only the numerical features
    input_scaled = input_df.copy()

    input_scaled[numeric_features] = scaler.transform(
        input_df[numeric_features]
    )

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts the positive HeartDisease class.")
    else:
        st.success("The model predicts the negative HeartDisease class.")

    st.caption(
        "Educational project only. This prediction is not a medical "
        "diagnosis and should not be used for medical decisions."
    )