
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder

# Download and load model from Hugging Face
model_path = hf_hub_download(repo_id="Vaddiritz/Tourism-Package-Prediction-rithika", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Recommendation App")
st.write("""
This application predicts whether a customer is likely to purchase a **tourism package** 
based on their profile and preferences.
Fill in the details below to get a prediction.
""")

# Customer Details
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=1)
preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
numberoftrips = st.number_input("Number Of Trips", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport", [0, 1])
owncar = st.selectbox("Own Car", [0, 1])
numberofchildrenvisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP"])
monthlyincome = st.number_input("Monthly Income", min_value=1000, value=50000)
pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Standard"])
numberoffollowups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=2)
durationofpitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=60, value=10)

# --- Create input dataframe ---
input_data = pd.DataFrame([[age, typeofcontact, citytier, occupation, gender,numberofpersonvisiting, preferredpropertystar, 
                            maritalstatus,numberoftrips, passport, owncar, numberofchildrenvisiting, designation, 
                            monthlyincome, pitchsatisfactionscore, productpitched,numberoffollowups, durationofpitch]], 
                          columns=["Age", "TypeofContact", "CityTier", "Occupation", "Gender",
                                   "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
                                   "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
                                   "Designation", "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched",
                                   "NumberOfFollowups", "DurationOfPitch"])


# Display input summary
st.subheader("Entered Details:")
st.write(input_data)

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase Package" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(result)
