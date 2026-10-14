
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("credit_scoring_model.pkl")

st.set_page_config(page_title="Credit Scoring", page_icon="💳")

st.title("💳 Credit Scoring Model")
st.write("Enter customer details to estimate credit risk.")

st.info(
    "Educational demonstration only. "
    "This prediction is not a real lending decision."
)

with st.form("credit_form"):
    st.subheader("Customer Information")

    checking_account = st.selectbox(
        "Checking account status",
        ["A11", "A12", "A13", "A14"]
    )

    duration = st.number_input(
        "Loan duration (months)", min_value=4, max_value=72, value=12
    )

    credit_history = st.selectbox(
        "Credit history", ["A30", "A31", "A32", "A33", "A34"]
    )

    purpose = st.selectbox(
        "Loan purpose",
        ["A40", "A41", "A42", "A43", "A44", "A45",
         "A46", "A47", "A48", "A49", "A410"]
    )

    credit_amount = st.number_input(
        "Credit amount", min_value=250, max_value=20000, value=2000
    )

    savings_account = st.selectbox(
        "Savings account status", ["A61", "A62", "A63", "A64", "A65"]
    )

    employment_since = st.selectbox(
        "Employment duration",
        ["A71", "A72", "A73", "A74", "A75"]
    )

    installment_rate = st.selectbox(
        "Installment rate (% of disposable income)", [1, 2, 3, 4]
    )

    personal_status_sex = st.selectbox(
        "Personal status and sex",
        ["A91", "A92", "A93", "A94", "A95"]
    )

    other_debtors = st.selectbox(
        "Other debtors or guarantors", ["A101", "A102", "A103"]
    )

    residence_since = st.selectbox(
        "Years at current residence", [1, 2, 3, 4]
    )

    property_type = st.selectbox(
        "Property type", ["A121", "A122", "A123", "A124"]
    )

    age = st.number_input("Age", min_value=19, max_value=75, value=30)

    other_installment_plans = st.selectbox(
        "Other installment plans", ["A141", "A142", "A143"]
    )

    housing = st.selectbox("Housing", ["A151", "A152", "A153"])

    existing_credits = st.selectbox(
        "Existing credits at this bank", [1, 2, 3, 4]
    )

    job = st.selectbox("Job category", ["A171", "A172", "A173", "A174"])

    dependents = st.selectbox("Number of dependents", [1, 2])

    telephone = st.selectbox("Telephone", ["A191", "A192"])

    foreign_worker = st.selectbox("Foreign worker", ["A201", "A202"])

    submitted = st.form_submit_button("Predict Credit Risk")

if submitted:
    customer = pd.DataFrame([{
        "checking_account": checking_account,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_account": savings_account,
        "employment_since": employment_since,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property_type,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "dependents": dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker
    }])

    prediction = int(model.predict(customer)[0])
    probabilities = model.predict_proba(customer)[0]

    # Class 0 = good credit; class 1 = bad credit
    bad_risk_probability = float(probabilities[1])

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Predicted category: Good credit risk")
    else:
        st.error("Predicted category: Bad credit risk")

    st.write(
        f"Model-estimated bad-risk probability: "
        f"{bad_risk_probability:.1%}"
    )

    st.caption(
        "This is a model estimate on historical data, not a verified "
        "probability of default or a recommendation to approve or reject "
        "a real applicant."
    )
