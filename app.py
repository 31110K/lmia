import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.title("Student Score Predictor")  # Or your app title

# --- Sidebar for inputs ---
st.sidebar.header("Enter Data for Prediction")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.sidebar.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
prep_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, value=50)

# --- Button to Predict ---
if st.button("Predict"):
    # Construct CustomData object
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parent_edu,
        lunch=lunch,
        test_preparation_course=prep_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert to dataframe
    pred_df = data.get_data_as_data_frame()
    st.write("Input Data:")
    st.dataframe(pred_df)

    # Predict
    pipeline = PredictPipeline()
    result = pipeline.predict(pred_df)
    st.success(f"Predicted Score: {min(99, result[0])}")
