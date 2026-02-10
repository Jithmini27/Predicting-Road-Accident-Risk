import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# Page Config

st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="🚦",
    layout="centered"
)

# Load Model

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# App Title

st.title("🚦 Road Accident Risk Prediction App")
st.markdown(
    """
    This web application predicts the likelihood of a road accident  
    based on road and environmental conditions.

    **Kaggle Competition:** Playground Series S5E10  
    """
)

st.divider()


# Sidebar Inputs

st.sidebar.header("📌 Enter Road Conditions")

road_type = st.sidebar.selectbox(
    "Road Type",
    ["Highway", "City Road", "Rural Road"]
)

lighting = st.sidebar.selectbox(
    "Lighting Condition",
    ["Daylight", "Night", "Dim"]
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Rainy", "Foggy", "Storm"]
)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

num_lanes = st.sidebar.slider("Number of Lanes", 1, 6, 2)

curvature = st.sidebar.slider("Road Curvature", 0.0, 1.0, 0.3)

speed_limit = st.sidebar.slider("Speed Limit (km/h)", 20, 120, 60)

num_reported_accidents = st.sidebar.slider(
    "Previously Reported Accidents",
    0, 50, 5
)

road_signs_present = st.sidebar.checkbox("Road Signs Present")

public_road = st.sidebar.checkbox("Public Road")

holiday = st.sidebar.checkbox("Holiday")

school_season = st.sidebar.checkbox("School Season")


# Convert Inputs into DataFrame

input_data = pd.DataFrame({
    "id": [0],  
    "road_type": [road_type],
    "num_lanes": [num_lanes],
    "curvature": [curvature],
    "speed_limit": [speed_limit],
    "lighting": [lighting],
    "weather": [weather],
    "road_signs_present": [road_signs_present],
    "public_road": [public_road],
    "time_of_day": [time_of_day],
    "holiday": [holiday],
    "school_season": [school_season],
    "num_reported_accidents": [num_reported_accidents]
})

# Ensure correct columns (drop target if accidentally present)
if "accident_risk" in input_data.columns:
    input_data = input_data.drop(columns=["accident_risk"])


# Prediction Button

st.subheader("🎯 Accident Risk Prediction")

if st.button("Predict Accident Risk 🚀"):

    prediction = model.predict(input_data)[0]
    prediction = np.clip(prediction, 0, 1)

    st.success(f"Predicted Accident Risk Score: **{prediction:.3f}**")

    # Risk Level Display
    if prediction < 0.33:
        st.info("🟢 Risk Level: LOW")
    elif prediction < 0.66:
        st.warning("🟠 Risk Level: MEDIUM")
    else:
        st.error("🔴 Risk Level: HIGH")

    st.progress(float(prediction))


# Footer

st.divider()
st.markdown(
    """
    **Developed by:** Jithmini Benara Kumbalatharaarachchi  
    Final Year Business Intelligence Project  
    """
)


