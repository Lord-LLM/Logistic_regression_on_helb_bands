"""
DSAIC Club - HELB Band Placement Predictor (Multinomial Logistic Regression)
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HELB Band Placement Predictor", page_icon="🎓")

@st.cache_resource
def load_model():
    return joblib.load("model/helb_band_pipeline.joblib")

pipeline = load_model()

st.title("HELB Band Placement Predictor")

occupations = ["formal_employment", "informal_business", "unemployed", "deceased_both", "deceased_one"]
orphan_statuses = ["not_orphan", "single_orphan", "double_orphan"]
residence_types = ["rural_high_poverty", "rural_low_poverty", "urban_informal", "urban_formal"]

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Household monthly income (KES)", 500, 300000, 15000, step=500)
    parent_occupation = st.selectbox("Parent/guardian occupation", occupations)
    orphan_status = st.selectbox("Orphan status", orphan_statuses)
    disability = st.checkbox("Living with a disability")
with col2:
    dependents_count = st.slider("Number of dependents in household", 0, 10, 3)
    siblings_in_college = st.slider("Siblings also in college/university", 0, 4, 0)
    residence_type = st.selectbox("Place of residence", residence_types)
    gender = st.selectbox("Gender", ["female", "male"])

if st.button("Predict band", type="primary"):
    X = pd.DataFrame([{
        "household_monthly_income_kes": income,
        "dependents_count": dependents_count,
        "siblings_in_college": siblings_in_college,
        "disability": int(disability),
        "parent_occupation": parent_occupation,
        "orphan_status": orphan_status,
        "residence_type": residence_type,
        "gender": gender,
    }])
    pred_band = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    classes = pipeline.classes_

    st.success(f"Predicted band: **Band {pred_band}**  "
               f"({'highest need' if pred_band == 1 else 'lowest need' if pred_band == 5 else 'moderate need'})")

    st.write("Predicted probability by band:")
    prob_df = pd.DataFrame({"Band": [f"Band {c}" for c in classes], "Probability": proba}).set_index("Band")
    st.bar_chart(prob_df)
    st.caption("This is a teaching estimate from a synthetic dataset. Never use it for real HELB applications.")
