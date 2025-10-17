import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
from pathlib import Path
import dotenv
dotenv.load_dotenv()  



def web_predict():
    st.title(" Bank Subscription Prediction")

    #  User Inputs 
    age = st.number_input("Age", 18, 100, 35) # default 35 , min 18, max 100
    job = st.selectbox("Job", ["student", "management", "technician", "admin.", "services", "retired", "self-employed", "entrepreneur", "unemployed", "housemaid", "blue-collar", "unknown"])
    marital = st.selectbox("Marital Status", ["single", "married", "divorced"]) # 3 options Single, Married, Divorced
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has credit in default?", ["no", "yes"])
    balance = st.number_input("Average Yearly Balance (€)", -5000, 100000, 500)
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    campaign = st.number_input("Number of Contacts in this Campaign", 1, 20, 2)
    pdays = st.number_input("Days since last contact (-1 if never)", -1, 999, -1)
    previous = st.number_input("Number of Previous Contacts", 0, 20, 0)
    poutcome = st.selectbox("Previous Outcome", ["success", "failure", "other", "unknown"])

    # initialize session variables only if they don't already exist
    if 'prob' not in st.session_state:
        st.session_state.prob = None
    if 'pred' not in st.session_state:
        st.session_state.pred = None
    

    if st.button("Predict"):
        model = st.session_state.get("model")
        model_path = r"c:\Github Projects\AI-MARKETING-STRATEGY_ADVISOR\models\final_model.pkl"
        model = load(model_path)
        st.session_state["model"] = model

        d = {
            "age": age, "job": job, "marital": marital, "education": education, "default": default,
            "balance": balance, "housing": housing, "loan": loan, "contact": contact,
            "day": 15, "month": month, "duration": 180, "campaign": campaign,
            "pdays": pdays, "previous": previous, "poutcome": poutcome, "y": "no"
        }
        df = pd.DataFrame([d])

        # preprocessing
        df["y"] = df["y"].map({"no": 0, "yes": 1})
        df["ever_contacted"] = np.where((df["pdays"] == -1) & (df["previous"] == 0), 0, 1)
        df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
        df["education_ord"] = df["education"].map({"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}).fillna(0).astype(int)
        df["campaign"] = df["campaign"].clip(1)
        df["previous"] = df["previous"].clip(0)
        X = df.drop(columns=["balance", "pdays", "education", "y"])
        
        st.session_state.pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        st.session_state.prob = round(float(probs[st.session_state.pred]) * 100, 2)
        

    if st.session_state.pred == 1:
        st.success(f"Prediction: YES — User Subscribed ({st.session_state.prob}%)")
    if st.session_state.pred == 0:
        st.error(f"Prediction: NO — User Did Not Subscribe ({st.session_state.prob}%)")

    st.caption("Model used: models/final_model.pkl")
    return st.session_state.prob, contact, previous, education, age, marital, loan
    


if __name__ == '__main__':
    web_predict()