
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import HfApi, hf_hub_download
import os
import re

# -------------------------------
# Hugging Face setup
# -------------------------------
hf_token = os.getenv("HF_TOKEN")
repo_id = "Vaddiritz/Tourism-Package-Prediction-rithika_new"
api = HfApi()

# Get list of files in repo and pick the latest versioned pipeline
all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
# Filter files matching 'tourism_pipeline_YYYYMMDD_HHMMSS.joblib'
pipeline_files = [f for f in all_files if re.match(r"tourism_pipeline.joblib", f)]
if not pipeline_files:
    st.error("No model pipeline found in Hugging Face repo.")
    st.stop()

# Sort and pick latest
latest_pipeline_file = sorted(pipeline_files)[-1]

# Download the latest pipeline
model_path = hf_hub_download(repo_id=repo_id, filename=latest_pipeline_file, repo_type="model")
model = joblib.load(model_path)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Tourism Package Recommendation App")
st.write("""
This app predicts whether a customer is likely to purchase a **tourism package** 
based on their profile and preferences.
Fill in the details below to get a prediction.
""")

# -------------------------------
# User input collection
# -------------------------------
def user_input_features():
  data = {
        "Age": st.number_input("Age", 18, 100, 30),
        "TypeofContact": st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"]),
        "CityTier": st.selectbox("City Tier", [1, 2, 3]),
        "Occupation": st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "NumberOfPersonVisiting": st.number_input("Number of Person Visiting", 1, 10, 1),
        "PreferredPropertyStar": st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5]),
        "MaritalStatus": st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"]),
        "NumberOfTrips": st.number_input("Number of Trips", 0, 20, 1),
        "Passport": st.selectbox("Passport", [0, 1]),
        "OwnCar": st.selectbox("Own Car", [0, 1]),
        "NumberOfChildrenVisiting": st.number_input("Number of Children Visiting", 0, 10, 0),
        "Designation": st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"]),
        "MonthlyIncome": st.number_input("Monthly Income", 1000, 1000000, 50000),
        "PitchSatisfactionScore": st.slider("Pitch Satisfaction Score", 1, 5, 3),
        "ProductPitched": st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]),
        "NumberOfFollowups": st.number_input("Number of Followups", 0, 20, 2),
        "DurationOfPitch": st.number_input("Duration Of Pitch (minutes)", 0, 60, 10)}
  return pd.DataFrame([data])


input_df = user_input_features()

st.subheader("Entered Details")
st.write(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Package Purchase"):
    try:
        # Pipeline automatically preprocesses categorical features
        # Extract preprocessor from the pipeline
        preprocessor = model.named_steps['columntransformer']
        # Transform input data
        X_processed = preprocessor.transform(input_df)
        # Get the classifier from the pipeline
        classifier = model.named_steps['xgbclassifier']
        # Make prediction
        prediction = classifier.predict(X_processed)[0]
        prediction_proba = classifier.predict_proba(X_processed)[0]
        #prediction = model.predict(input_df)[0]
        #probability = model.predict_proba(input_df)[0][1]
        #result = "Likely to Purchase Package" if prediction == 1 else "Unlikely to Purchase"
        #st.subheader("Prediction Result")
        #st.success(result)
        #st.info(f"Probability of purchasing: {probability:.2f}")
        # Display results
        st.subheader("Prediction Result:")
        if prediction == 1:
          st.success("The customer is **likely to purchase** the tourism package!")
        else:
          st.warning("The customer is **unlikely to purchase** the tourism package.")

        st.write(f"**Probability of Purchase**: {prediction_proba[1]:.2%}")
        st.write(f"**Probability of No Purchase**: {prediction_proba[0]:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
