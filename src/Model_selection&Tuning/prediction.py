import numpy as np, pandas as pd
from joblib import load


def run():
    print("\n=== Bank Marketing Subscription Prediction ===\n")

    # simple user input (case-insensitive)
    age = int(input("Enter Age: "))
    job = input("Enter Job (student/management/technician/admin./services/retired/self-employed/entrepreneur/unemployed/housemaid/blue-collar/unknown): ").strip().lower() 
    marital = input("Enter Marital Status (single/married/divorced): ").strip().lower()
    education = input("Enter Education (primary/secondary/tertiary/unknown): ").strip().lower() 
    default = input("Has credit in default? (yes/no): ").strip().lower()
    balance = int(input("Enter Average Yearly Balance (in euros): "))
    housing = input("Has Housing Loan? (yes/no): ").strip().lower()
    loan = input("Has Personal Loan? (yes/no): ").strip().lower()
    contact = input("Contact Type (cellular/telephone/unknown): ").strip().lower()
    month = input("Last Contact Month (jan..dec): ").strip().lower()[0:3]
    campaign = int(input("Number of Contacts in this Campaign: "))
    pdays = int(input("Days since last contact (-1 if never): "))
    previous = int(input("Number of Previous Contacts: "))
    poutcome = input("Previous Outcome (success/failure/other/unknown): ").strip().lower()

    # small corrections
    if "self" in job: 
        job = "self-employed"

    if "blue" in job: 
        job = "blue-collar"

    if "house" in job: 
        job = "housemaid"

    if job == "admin": 
        job = "admin."

    if job not in {"blue-collar","management","technician","admin.","services","retired","self-employed","entrepreneur","unemployed","housemaid","student","unknown"}:
        job = "unknown"

    if education in {"graduate","bachelor","master","phd"}: 
        education = "tertiary"

    elif education in {"highschool","high school"}: 
        education = "secondary"

    elif education not in {"primary","secondary","tertiary","unknown"}: 
        education = "unknown"

    if not month[:3] in {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"}:
        month = "may"

   
    if contact.startswith("tel"): contact = "telephone"
    elif contact.startswith("cel"): contact = "cellular"
    else: contact = "unknown"

    if poutcome not in {"success","failure","other","unknown"}:
        poutcome = "unknown"

    # raw-like dataframe
    df = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education, "default": default,
        "balance": balance, "housing": housing, "loan": loan, "contact": contact,
        "day": 15, "month": month, "duration": 180, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome, "y": "no"
    }])

    # replicate preprocessing
    df["y"] = df["y"].map({"no":0,"yes":1})
    df["ever_contacted"] = np.where((df["pdays"] == -1) & (df["previous"] == 0), 0, 1)
    df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["education_ord"] = df["education"].map({"unknown":0,"primary":1,"secondary":2,"tertiary":3}).fillna(0).astype(int)
    df["campaign"] = df["campaign"].clip(1)
    df["previous"] = df["previous"].clip(0)
    X = df.drop(columns=["balance","pdays","education","y"])

    # load model
    
    model = load("models/final_model.pkl")

    # predict

    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0] # used predict_proba to get probabilities of each class ,[0] to get first row
    prob = round(float(probs[pred]) * 100, 2)

    # Output 
    print("\===============================")
    if pred == 1:
        print(f" Prediction: YES — User Subscribed   (Probability = {prob}%)")
    else:
        print(f" Prediction: NO — User Did Not Subscribe   (Probability = {prob}%)")
    print("===============================/")

if __name__ == "__main__":
    run()