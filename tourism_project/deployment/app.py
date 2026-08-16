import streamlit as st
import pandas as pd
import joblib

st.title("🌴 Wellness Tourism Package Purchase Predictor")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("tourism_project/deployment/best_model.pkl")

model = load_model()

st.sidebar.header("Customer Profile Input")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.sidebar.number_input("Pitch Duration (mins)", min_value=1, max_value=120, value=15)
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
person_visiting = st.sidebar.slider("Persons Visiting", 1, 10, 2)
followups = st.sidebar.slider("Follow-ups", 1, 10, 3)
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_star = st.sidebar.selectbox("Preferred Hotel Star", [3, 4, 5])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
trips = st.sidebar.number_input("Annual Trips", min_value=0, max_value=50, value=3)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": person_visiting,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": trips
}])

st.subheader("Submitted Input Profile")
st.dataframe(input_data)

if st.button("Predict Likelihood"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None
        
        if prediction[0] == 1:
            st.success(f"🎉 Customer is likely to purchase! (Confidence: {probability:.2%})")
        else:
            st.warning(f"❌ Customer is unlikely to purchase. (Confidence: {1 - probability:.2%})")
    except Exception as e:
        st.error(f"Error predicting outcome: {e}")
