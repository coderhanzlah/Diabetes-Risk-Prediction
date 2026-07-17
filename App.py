# ==========================================================
# Diabetes Risk Prediction System
# Materials Modelling & Simulation Laboratory
# Department of Physics
# Allama Iqbal Open University
# Islamabad, Pakistan
# ==========================================================

import os

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# Page Configuration
# This must be the first Streamlit command.
# ==========================================================

st.set_page_config(
    page_title=(
        "Diabetes Risk Prediction | Materials Modelling & Simulation Lab "
        "| Department of Physics | AIOU"
    ),
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# Project Information
# ==========================================================

PROJECT_TITLE = "Diabetes Risk Prediction System"

LAB_NAME = "Materials Modelling & Simulation Laboratory"

DEPARTMENT = "Department of Physics"

UNIVERSITY = "Allama Iqbal Open University"

LOCATION = "Islamabad, Pakistan"

PROJECT_DESCRIPTION = (
    "Artificial Intelligence and Machine Learning-Based "
    "Clinical Decision Support Tool"
)


# ==========================================================
# File Paths
# ==========================================================

BASE_FOLDER = r"D:\Machine Learning\Diabetes"

MODEL_PATH = os.path.join(
    BASE_FOLDER,
    "diabetes_xgboost_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_FOLDER,
    "diabetes_scaler.pkl"
)

LOGO_PATH = os.path.join(
    BASE_FOLDER,
    "aiou_logo.png"
)


# ==========================================================
# Feature Configuration
# The order must match the model training feature order.
# ==========================================================

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


# ==========================================================
# Custom Styling
# ==========================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------
       Main application background
    ------------------------------------------------------ */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #eef6ff 0%,
                #ffffff 48%,
                #eefaf6 100%
            );
    }

    .block-container {
        max-width: 1050px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ------------------------------------------------------
       Remove unnecessary Streamlit spacing
    ------------------------------------------------------ */

    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    /* ------------------------------------------------------
       Institutional header
    ------------------------------------------------------ */

    .institution-header {
        background:
            linear-gradient(
                120deg,
                #022b4f,
                #075985,
                #0f766e
            );

        padding: 24px 20px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(2, 43, 79, 0.18);
    }

    .institution-header h1 {
        color: #ffd54f;
        font-size: 30px;
        font-weight: 750;
        margin: 0 0 7px 0;
    }

    .institution-header h2 {
        color: white;
        font-size: 23px;
        margin: 3px 0;
        font-weight: 650;
    }

    .institution-header h3 {
        color: #e0f2fe;
        font-size: 20px;
        margin: 3px 0;
        font-weight: 550;
    }

    .institution-header p {
        color: #dbeafe;
        font-size: 16px;
        margin: 6px 0 0 0;
    }

    /* ------------------------------------------------------
       Application title
    ------------------------------------------------------ */

    .application-title {
        text-align: center;
        padding: 10px 10px 16px 10px;
    }

    .application-title h1 {
        color: #063b64;
        font-size: 38px;
        font-weight: 780;
        margin: 0 0 8px 0;
    }

    .application-title p {
        color: #475569;
        font-size: 18px;
        margin: 0;
    }

    /* ------------------------------------------------------
       Introduction box
    ------------------------------------------------------ */

    .intro-box {
        background-color: rgba(255, 255, 255, 0.92);
        border-left: 5px solid #0f766e;
        padding: 16px 18px;
        border-radius: 11px;
        margin: 12px 0 16px 0;
        box-shadow: 0 3px 12px rgba(15, 118, 110, 0.08);
        color: #334155;
        font-size: 16px;
    }

    /* ------------------------------------------------------
       Section headings
    ------------------------------------------------------ */

    .section-heading {
        color: #063b64;
        font-size: 24px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 2px solid #0f766e;
    }

    /* ------------------------------------------------------
       Metric cards
    ------------------------------------------------------ */

    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.94);
        border: 1px solid #cbd5e1;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #063b64;
        font-weight: 750;
    }

    /* ------------------------------------------------------
       Input fields
    ------------------------------------------------------ */

    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    input {
        font-size: 16px !important;
    }

    label {
        font-weight: 600 !important;
        color: #1e293b !important;
    }

    /* ------------------------------------------------------
       Prediction button
    ------------------------------------------------------ */

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background:
            linear-gradient(
                90deg,
                #075985,
                #0f766e
            );

        color: white;
        border: none;
        border-radius: 11px;
        min-height: 52px;
        font-size: 18px;
        font-weight: 700;
        transition: 0.2s ease-in-out;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background:
            linear-gradient(
                90deg,
                #0c4a6e,
                #115e59
            );

        color: white;
        border: none;
    }

    /* ------------------------------------------------------
       Prediction result boxes
    ------------------------------------------------------ */

    .risk-high {
        background-color: #fff1f2;
        border: 2px solid #e11d48;
        color: #881337;
        padding: 20px;
        border-radius: 13px;
        text-align: center;
        font-size: 21px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 14px;
    }

    .risk-low {
        background-color: #ecfdf5;
        border: 2px solid #059669;
        color: #065f46;
        padding: 20px;
        border-radius: 13px;
        text-align: center;
        font-size: 21px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 14px;
    }

    /* ------------------------------------------------------
       Footer
    ------------------------------------------------------ */

    .developer-footer {
        background:
            linear-gradient(
                120deg,
                #022b4f,
                #075985,
                #0f766e
            );

        padding: 24px 18px;
        border-radius: 17px;
        text-align: center;
        color: white;
        margin-top: 32px;
        box-shadow: 0 8px 24px rgba(2, 43, 79, 0.18);
    }

    .developer-footer .developer-label {
        color: #ffd54f;
        font-size: 17px;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .developer-footer h2 {
        color: white;
        font-size: 23px;
        margin: 5px 0;
    }

    .developer-footer h3 {
        color: #e0f2fe;
        font-size: 18px;
        margin: 4px 0;
    }

    .developer-footer p {
        color: #dbeafe;
        font-size: 15px;
        margin: 5px 0;
    }

    .developer-footer hr {
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.28);
        margin: 16px 0;
    }

    /* ------------------------------------------------------
       Expander
    ------------------------------------------------------ */

    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        border: 1px solid #dbe3ec;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Institutional Header
# ==========================================================

if os.path.isfile(LOGO_PATH):

    logo_column, header_column = st.columns(
        [1, 5],
        vertical_alignment="center"
    )

    with logo_column:
        st.image(
            LOGO_PATH,
            width=115
        )

    with header_column:
        st.markdown(
            f"""
            <div class="institution-header">
                <h1>{LAB_NAME}</h1>
                <h2>{DEPARTMENT}</h2>
                <h3>{UNIVERSITY}</h3>
                <p>{LOCATION}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

else:

    st.markdown(
        f"""
        <div class="institution-header">
            <h1>{LAB_NAME}</h1>
            <h2>{DEPARTMENT}</h2>
            <h3>{UNIVERSITY}</h3>
            <p>{LOCATION}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# Application Title
# ==========================================================

st.markdown(
    f"""
    <div class="application-title">
        <h1>🩺 {PROJECT_TITLE}</h1>
        <p>{PROJECT_DESCRIPTION}</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Load Model and Scaler
# ==========================================================

@st.cache_resource
def load_model_files():
    """
    Load the trained machine learning model and preprocessing scaler.
    """

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found:\n{SCALER_PATH}"
        )

    loaded_model = joblib.load(MODEL_PATH)

    loaded_scaler = joblib.load(SCALER_PATH)

    if not hasattr(loaded_model, "predict"):
        raise TypeError(
            "The loaded model does not contain a predict() method."
        )

    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError(
            "The loaded model does not contain a predict_proba() method."
        )

    if not hasattr(loaded_scaler, "transform"):
        raise TypeError(
            "The loaded scaler does not contain a transform() method."
        )

    return loaded_model, loaded_scaler


try:

    model, scaler = load_model_files()

except Exception as error:

    st.error(
        "The trained model or scaler could not be loaded."
    )

    st.code(str(error))

    st.info(
        "Confirm that the following files exist:\n\n"
        f"Model: {MODEL_PATH}\n\n"
        f"Scaler: {SCALER_PATH}"
    )

    st.stop()


# ==========================================================
# Application Introduction
# ==========================================================

st.markdown(
    """
    <div class="intro-box">
        Enter the patient's clinical measurements to estimate the
        probability of diabetes using a trained
        <strong>XGBoost machine learning model</strong>.
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "This application is intended only for educational and research "
    "purposes. It must not replace medical examination, laboratory "
    "testing, clinical diagnosis, or consultation with a qualified "
    "healthcare professional."
)


# ==========================================================
# Model Summary Cards
# ==========================================================

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric(
        label="Model",
        value="XGBoost"
    )

with summary_col2:
    st.metric(
        label="Input Features",
        value="8"
    )

with summary_col3:
    st.metric(
        label="Test Accuracy",
        value="75.3%"
    )

with summary_col4:
    st.metric(
        label="ROC-AUC",
        value="0.827"
    )


# ==========================================================
# Patient Input Form
# ==========================================================

st.markdown(
    '<div class="section-heading">👤 Patient Clinical Information</div>',
    unsafe_allow_html=True
)

with st.form(
    key="diabetes_prediction_form",
    clear_on_submit=False
):

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        pregnancies = st.number_input(
            label="Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Number of previous pregnancies."
        )

        glucose = st.number_input(
            label="Glucose concentration (mg/dL)",
            min_value=1.0,
            max_value=300.0,
            value=120.0,
            step=1.0,
            format="%.1f",
            help="Plasma glucose concentration."
        )

        blood_pressure = st.number_input(
            label="Diastolic blood pressure (mmHg)",
            min_value=1.0,
            max_value=200.0,
            value=72.0,
            step=1.0,
            format="%.1f",
            help="Diastolic blood pressure."
        )

        skin_thickness = st.number_input(
            label="Skin thickness (mm)",
            min_value=1.0,
            max_value=100.0,
            value=29.0,
            step=1.0,
            format="%.1f",
            help="Triceps skin-fold thickness."
        )

    with input_col2:

        insulin = st.number_input(
            label="Insulin level (μU/mL)",
            min_value=1.0,
            max_value=1000.0,
            value=125.0,
            step=1.0,
            format="%.1f",
            help="Two-hour serum insulin measurement."
        )

        bmi = st.number_input(
            label="Body mass index (kg/m²)",
            min_value=1.0,
            max_value=80.0,
            value=30.0,
            step=0.1,
            format="%.1f",
            help="Body mass index calculated from height and weight."
        )

        pedigree = st.number_input(
            label="Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.470,
            step=0.010,
            format="%.3f",
            help=(
                "A dataset-derived numerical indicator of hereditary "
                "diabetes risk."
            )
        )

        age = st.number_input(
            label="Age (years)",
            min_value=1,
            max_value=120,
            value=35,
            step=1,
            help="Patient age in years."
        )

    predict_button = st.form_submit_button(
        label="🔍 Predict Diabetes Risk",
        type="primary",
        use_container_width=True
    )


# ==========================================================
# Prediction
# ==========================================================

if predict_button:

    patient_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]],
        columns=FEATURE_COLUMNS
    )

    try:

        patient_scaled = scaler.transform(
            patient_data
        )

        prediction = int(
            model.predict(patient_scaled)[0]
        )

        probabilities = model.predict_proba(
            patient_scaled
        )[0]

        if len(probabilities) < 2:
            raise ValueError(
                "The model returned fewer than two class probabilities."
            )

        non_diabetic_probability = float(
            probabilities[0]
        )

        diabetic_probability = float(
            probabilities[1]
        )

    except Exception as error:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.code(str(error))

        st.stop()

    st.divider()

    st.markdown(
        '<div class="section-heading">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            label="Estimated Diabetes Probability",
            value=f"{diabetic_probability * 100:.1f}%"
        )

    with result_col2:

        st.metric(
            label="Estimated Non-Diabetes Probability",
            value=f"{non_diabetic_probability * 100:.1f}%"
        )

    st.write("#### Diabetes risk probability")

    st.progress(
        min(
            max(
                diabetic_probability,
                0.0
            ),
            1.0
        )
    )

    if prediction == 1:

        st.markdown(
            """
            <div class="risk-high">
                ⚠️ Higher Predicted Diabetes Risk
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "The entered clinical measurements were classified by the "
            "machine learning model as being associated with a comparatively "
            "higher probability of diabetes."
        )

    else:

        st.markdown(
            """
            <div class="risk-low">
                ✅ Lower Predicted Diabetes Risk
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "The entered clinical measurements were classified by the "
            "machine learning model as being associated with a comparatively "
            "lower probability of diabetes."
        )

    st.write("#### Class Probability Distribution")

    probability_table = pd.DataFrame(
        {
            "Classification": [
                "Non-diabetic",
                "Diabetic"
            ],
            "Probability": [
                f"{non_diabetic_probability * 100:.2f}%",
                f"{diabetic_probability * 100:.2f}%"
            ]
        }
    )

    st.dataframe(
        probability_table,
        hide_index=True,
        use_container_width=True
    )

    st.write("#### Entered Patient Measurements")

    display_data = patient_data.rename(
        columns={
            "BloodPressure": "Blood Pressure",
            "SkinThickness": "Skin Thickness",
            "DiabetesPedigreeFunction": "Pedigree Function"
        }
    )

    st.dataframe(
        display_data,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "The displayed probability is a statistical prediction generated "
        "by the trained model and is not a confirmed medical diagnosis. "
        "A proper diabetes assessment requires clinical examination, "
        "appropriate laboratory testing, and interpretation by a qualified "
        "healthcare professional."
    )


# ==========================================================
# Model Information
# ==========================================================

with st.expander(
    "ℹ️ Model Information and Performance"
):

    st.markdown("### Machine Learning Model")

    st.write(
        "**Algorithm:** XGBoost classifier"
    )

    st.write(
        "**Number of input features:** 8"
    )

    st.write(
        "**Input variables:** Pregnancies, glucose concentration, "
        "blood pressure, skin thickness, insulin level, body mass "
        "index, Diabetes Pedigree Function, and age."
    )

    st.markdown("### Reported Test Performance")

    performance_table = pd.DataFrame(
        {
            "Evaluation Metric": [
                "Test Accuracy",
                "ROC-AUC",
                "Recall for Diabetic Class"
            ],
            "Reported Value": [
                "Approximately 75.3%",
                "Approximately 0.827",
                "Approximately 63%"
            ]
        }
    )

    st.dataframe(
        performance_table,
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "The diabetic-class recall indicates that the model may fail "
        "to identify some diabetic cases. Therefore, this application "
        "must not be used as an independent diagnostic or medical "
        "screening system."
    )


# ==========================================================
# Developer and Institutional Footer
# ==========================================================

st.markdown(
    f"""
    <div class="developer-footer">

        <div class="developer-label">
            Developed by
        </div>

        <h2>{LAB_NAME}</h2>

        <h3>{DEPARTMENT}</h3>

        <h3>{UNIVERSITY}</h3>

        <p>{LOCATION}</p>

        <hr>

        <p>
            Artificial Intelligence • Machine Learning •
            Healthcare Analytics • Computational Research
        </p>

        <p>
            Developed for educational and research purposes only.
        </p>

        <p>
            © 2026 {UNIVERSITY}. All rights reserved.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)