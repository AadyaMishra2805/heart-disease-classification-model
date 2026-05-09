import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

html, body, [class*="css"] {
    overflow: hidden;
}

h1 {
    text-align: center;
    color: #ff4b4b;
    font-size: 42px !important;
    margin-bottom: 0px;
}

.subtext {
    text-align: center;
    color: #9e9e9e;
    margin-bottom: 25px;
    font-size: 16px;
}

div.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 12px;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border: none;
    margin-top: 8px;
}

div.stButton > button:hover {
    background-color: #ff2e2e;
    color: white;
}

.stSelectbox label,
.stSlider label {
    font-size: 15px !important;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1>❤️ Heart Disease Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtext'>Machine Learning Based Health Prediction System</p>",
    unsafe_allow_html=True
)

# Layout columns
col1, col2, col3, col4 = st.columns(4)

# Column 1
with col1:

    age = st.slider("Age", 1, 100, 25)

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    cp = st.selectbox(
        "Chest Pain",
        [0, 1, 2, 3]
    )

# Column 2
with col2:

    trestbps = st.slider(
        "Blood Pressure",
        80, 200, 120
    )

    chol = st.slider(
        "Cholesterol",
        100, 600, 200
    )

    fbs = st.selectbox(
        "Blood Sugar",
        [0, 1]
    )

# Column 3
with col3:

    restecg = st.selectbox(
        "Rest ECG",
        [0, 1, 2]
    )

    thalach = st.slider(
        "Heart Rate",
        60, 220, 150
    )

    exang = st.selectbox(
        "Angina",
        [0, 1]
    )

# Column 4
with col4:

    oldpeak = st.slider(
        "Oldpeak",
        0.0, 6.0, 1.0
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal",
        [0, 1, 2, 3]
    )

# Convert sex
sex_value = 1 if sex == "Male" else 0

# Predict button
predict_button = st.button("Predict")

# Prediction
if predict_button:

    input_data = np.array([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    # Result
    if prediction[0] == 1:

        st.error("⚠️ High Risk of Heart Disease")

    else:

        st.success("✅ Low Risk of Heart Disease")