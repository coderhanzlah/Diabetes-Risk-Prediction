from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# Page configuration
# ==========================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction | AIOU",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# Portable file paths
# ==========================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "diabetes_xgboost_model.pkl"
SCALER_PATH = APP_DIR / "diabetes_scaler.pkl"
LOGO_PATH = APP_DIR / "logo.png"


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
# Styling
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef6ff, #ffffff, #eefaf6);
    }

    .block-container {
        max-width: 950px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .header-box {
        background: linear-gradient(120deg, #022b4f, #075985, #0f766e);
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }

    .header-box h1 {
        color: #ffd54f;
        margin-bottom: 8px;
    }

    .header-box h2,
    .header-box h3,
    .header-box p {
        color: white;
        margin: 4px;
    }

    .result-high {
        background: #fff1f2;
        border: 2px solid #e11d48;
        color: #881337;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 21px;
        font-weight: bold;
    }

    .result-low {
        background: #ecfdf5;
        border: 2px solid #059669;
        color: #065f46;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 21px;
        font-weight: bold;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #075985, #0f766e);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# Header
# ==========================================================

if LOGO_PATH.is_file():
    logo_col, title_col = st.columns([1, 5])

    with logo_col:
        st.image(str(LOGO_PATH), width=110)

    with title_col:
        st.markdown(
            """
            <div class="header-box">
                <h1>Materials Modelling & Simulation Laboratory</h1>
                <h2>Department of Physics</h2>
                <h3>Allama Iqbal Open University</h3>
                <p>Islamabad, Pakistan</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="header-box">
            <h1>Materials Modelling & Simulation Laboratory</h1>
            <h2>Department of Physics</h2>
            <h3>Allama Iqbal Open University</h3>
            <p>Islamabad, Pakistan</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("🩺 Diabetes Risk Prediction System")

st.write(
    "Enter the patient's clinical measurements to estimate diabetes risk "
    "using a trained XGBoost machine-learning model."
)

st.warning(
    "This application is intended only for educational and research purposes. "
    "It is not a substitute for medical diagnosis or professional consultation."
)


# ==========================================================
# Load model and scaler
# ==========================================================

@st.cache_resource
def load_files():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH.name}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:
    model, scaler = load_files()

except Exception as error:
    st.error("The model or scaler could not be loaded.")
    st.exception(error)
    st.stop()


# ==========================================================
# Model information
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "XGBoost")

with col2:
    st.metric("Features", "8")

with col3:
    st.metric("Accuracy", "75.3%")

with col4:
    st.metric("ROC-AUC", "0.827")


# ==========================================================
# Input form
# ==========================================================

st.subheader("Patient Clinical Information")

with st.form("diabetes_form"):
    left_col, right_col = st.columns(2)

    with left_col:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

        glucose = st.number_input(
            "Glucose concentration (mg/dL)",
            min_value=1.0,
            max_value=300.0,
            value=120.0,
            step=1.0,
        )

        blood_pressure = st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=1.0,
            max_value=200.0,
            value=72.0,
            step=1.0,
        )

        skin_thickness = st.number_input(
            "Skin thickness (mm)",
            min_value=1.0,
            max_value=100.0,
            value=29.0,
            step=1.0,
        )

    with right_col:
        insulin = st.number_input(
            "Insulin level (μU/mL)",
            min_value=1.0,
            max_value=1000.0,
            value=125.0,
            step=1.0,
        )

        bmi = st.number_input(
            "Body mass index (kg/m²)",
            min_value=1.0,
            max_value=80.0,
            value=30.0,
            step=0.1,
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.470,
            step=0.010,
            format="%.3f",
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=35,
            step=1,
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Risk",
        type="primary",
        use_container_width=True,
    )


# ==========================================================
# Prediction
# ==========================================================

if submitted:
    patient_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age,
        ]],
        columns=FEATURE_COLUMNS,
    )

    try:
        scaled_data = scaler.transform(patient_data)

        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]

        non_diabetic_probability = float(probabilities[0])
        diabetic_probability = float(probabilities[1])

    except Exception as error:
        st.error("An error occurred during prediction.")
        st.exception(error)
        st.stop()

    st.divider()
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Diabetes Probability",
            f"{diabetic_probability * 100:.1f}%",
        )

    with result_col2:
        st.metric(
            "Non-Diabetes Probability",
            f"{non_diabetic_probability * 100:.1f}%",
        )

    st.progress(
        float(min(max(diabetic_probability, 0.0), 1.0))
    )

    if prediction == 1:
        st.markdown(
            """
            <div class="result-high">
                ⚠️ Higher Predicted Diabetes Risk
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-low">
                ✅ Lower Predicted Diabetes Risk
            </div>
            """,
            unsafe_allow_html=True,
        )

    probability_table = pd.DataFrame(
        {
            "Classification": [
                "Non-diabetic",
                "Diabetic",
            ],
            "Probability": [
                f"{non_diabetic_probability * 100:.2f}%",
                f"{diabetic_probability * 100:.2f}%",
            ],
        }
    )

    st.subheader("Class Probability Distribution")

    st.dataframe(
        probability_table,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Entered Patient Measurements")

    st.dataframe(
        patient_data,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "This prediction is a statistical estimate and not a confirmed "
        "medical diagnosis."
    )


# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#475569;">
        <strong>Developed by the Materials Modelling & Simulation Laboratory</strong><br>
        Department of Physics, Allama Iqbal Open University<br>
        Islamabad, Pakistan<br><br>
        Developed for educational and research purposes only.
    </div>
    """,
    unsafe_allow_html=True,
)
