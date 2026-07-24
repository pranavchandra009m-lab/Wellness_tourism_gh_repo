
import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# ==============================
# Load Model from Hugging Face
# ==============================

@st.cache_resource
def load_model():

    hf_model_repo_id = "Jags99/wellness-tourism-model"
    model_joblib_name = "wellness_tourism_v1.joblib"

    model_path = hf_hub_download(
        repo_id=hf_model_repo_id,
        filename=model_joblib_name
    )

    model = joblib.load(model_path)

    return model


wellness_tourism_model = load_model()


# ==============================
# Streamlit App
# ==============================

st.title("Wellness Tourism Prediction App")

st.write("Holaaaaaaaaaa...... Enter customer details to predict purchase probability.")


Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration of Pitch",
    min_value=0.0,
    max_value=300.0,
    value=15.0
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

NumberOfFollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    max_value=10,
    value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3.0, 4.0, 5.0]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0.0,
    max_value=20.0,
    value=3.0
)

Passport = st.selectbox(
    "Has Passport",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

OwnCar = st.selectbox(
    "Owns Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000.0,
    max_value=1000000.0,
    value=50000.0
)


# ==============================
# Create Input DataFrame
# ==============================

input_data = pd.DataFrame({

    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "ProductPitched": [ProductPitched],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome]

})


# ==============================
# Prediction
# ==============================

classification_threshold = 0.45


if st.button("Predict"):

    prediction_probability = (
        wellness_tourism_model.predict_proba(input_data)[0,1]
    )

    prediction = (
        prediction_probability >= classification_threshold
    )

    if prediction:
        st.success(
            f"Customer is likely to purchase the Wellness Tourism Package "
            f"(Probability: {prediction_probability:.2%})"
        )

    else:
        st.warning(
            f"Customer is unlikely to purchase the Wellness Tourism Package "
            f"(Probability: {prediction_probability:.2%})"
        )
