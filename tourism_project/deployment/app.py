import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide")
st.title("🌴 Wellness Tourism Package Purchase Predictor")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
    st.sidebar.header("Customer Profile Input")

    def user_inputs():
        age = st.sidebar.number_input("Age", 18, 80, 35)
        typeofcontact = st.sidebar.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
        citytier = st.sidebar.selectbox("City Tier", [1, 2, 3])
        duration = st.sidebar.number_input("Pitch Duration (mins)", 1, 120, 15)
        occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        persons = st.sidebar.slider("Persons Visiting", 1, 10, 2)
        followups = st.sidebar.slider("Follow-ups", 1, 10, 3)
        product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_star = st.sidebar.selectbox("Preferred Hotel Star", [3, 4, 5])
        marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        trips = st.sidebar.number_input("Annual Trips", 0, 30, 3)
        passport = st.sidebar.selectbox("Passport Owned", [0, 1])
        pitch_score = st.sidebar.slider("Pitch Score", 1, 5, 3)
        own_car = st.sidebar.selectbox("Owns Car", [0, 1])
        children = st.sidebar.slider("Children Visiting", 0, 5, 0)
        designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.sidebar.number_input("Monthly Income ($)", 1000, 100000, 25000)

        data = {
            'Age': age, 'TypeofContact': typeofcontact, 'CityTier': citytier,
            'DurationOfPitch': duration, 'Occupation': occupation, 'Gender': gender,
            'NumberOfPersonVisiting': persons, 'NumberOfFollowups': followups,
            'ProductPitched': product_pitched, 'PreferredPropertyStar': preferred_star,
            'MaritalStatus': marital_status, 'NumberOfTrips': trips, 'Passport': passport,
            'PitchSatisfactionScore': pitch_score, 'OwnCar': own_car,
            'NumberOfChildrenVisiting': children, 'Designation': designation,
            'MonthlyIncome': income
        }
        return pd.DataFrame([data])

    input_df = user_inputs()
    st.subheader("Submitted Input Profile")
    st.dataframe(input_df)

    if st.button("Predict Likelihood"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.success(f"🎯 **High Likelihood to Purchase** (Probability: {prob:.1%})")
        else:
            st.error(f"⚠️ **Low Likelihood to Purchase** (Probability: {prob:.1%})")

except Exception as e:
    st.error(f"Error loading model: {e}")
