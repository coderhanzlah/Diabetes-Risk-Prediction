from pathlib import Path

app_code = r'''# ==========================================================
# Diabetes Risk Prediction System
# Materials Modelling & Simulation Laboratory
# Department of Physics
# Allama Iqbal Open University
# Islamabad, Pakistan
# ==========================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# Page configuration
# This must be the first Streamlit command.
# ==========================================================

st.set_page_config(
    page_title=(
        "Diabetes Risk Prediction | Materials Modelling & Simulation Lab "
        "| Department of Physics | AIOU"
    ),
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# Project information
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
# Portable file paths
# These paths work locally and on Streamlit Community Cloud.
# ==========================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "diabetes_xgboost_model.pkl"
SCALER_PATH = APP_DIR / "diabetes_scaler.pkl"
LOGO_PATH = APP_DIR / "logo.png"


# ==========================================================
# Feature configuration
# The order must exactly match the model-training feature order.
# ==========================================================

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# ==========================================================
# Custom styling
# ==========================================================

st.markdown(
    """
    <style>
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

    header[data-testid="stHeader"] {
        background-color: transparent;
    }

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

    .intro-box {
        background-color: rgba(255, 255, 255, 0.92);
        border-left: 5px solid #0f766e;
        padding: 16px 18px;
        border-radius: 11px;
        margin: 12px 0 16px 0
