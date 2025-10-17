from pathlib import Path
import numpy as np, pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # we are using imbpipeline because it supports SMOTE in pipeline unlike sklearn pipeline in which we need to handle SMOTE separately.
# imblearn: SMOTE + pipeline (SMOTE runs only in fit, not in predict)

RAW_PATH   = "data/raw/bank-full.csv"   
PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models")
SEED = 42

# feature engineering on RAW data
def clean(df):
    cols = ["age","job","marital","education","balance","housing","loan","contact",
            "month","campaign","pdays","previous","poutcome","y"]
    df = df[cols].copy()

    # target
    df["y"] = df["y"].map({"no":0, "yes":1}).astype(int)

    # engineered features
    df["ever_contacted"] = np.where((df["pdays"] == -1) & (df["previous"] == 0), 0, 1) # out of 2 columns pdays and previous, we made a new column ever_contacted which tells if customer was ever contacted before or not.(how it works: if pdays is -1 and previous is 0 means never contacted before, else contacted before)
    df["balance_log"]    = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    df["education_ord"]  = df["education"].map({"unknown":0,"primary":1,"secondary":2,"tertiary":3}).fillna(0).astype(int)

    # light caps
    df["campaign"] = pd.to_numeric(df["campaign"], errors='coerce').fillna(0).astype(int).clip(upper=10)
    df["previous"] = pd.to_numeric(df["previous"], errors='coerce').fillna(0).astype(int).clip(upper=10)

    # drop replaced columns
    return df.drop(columns=["balance","pdays","education"])

def preprocessor(df_for_cols):
    num = ["age","campaign","previous","ever_contacted","balance_log","education_ord"]
    cat = ["job","marital","housing","loan","contact","month","poutcome"]
    return ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat), # sparse_output=False to get dense array output
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

# train on FULL data with SMOTE and save a single pipeline 
def build_full_pipeline_with_smote(raw_path):
    df_raw = pd.read_csv(raw_path, sep=";")
    df = clean(df_raw)
    df.to_csv(PROC_DIR/"processed_full.csv", index=False)

    X, y = df.drop(columns=["y"]), df["y"].astype(int)

    pipe = ImbPipeline(steps=[
        ("pre",   preprocessor(df)),          # encode/scale
        ("smote", SMOTE(random_state=SEED)),   # oversample minority 
        ("clf",   ExtraTreesClassifier(
            n_estimators=800, class_weight='balanced', random_state=42, n_jobs=-1
        )),
    ])

    pipe.fit(X, y)
    out = MODELS_DIR / "final_model.pkl"
    dump(pipe, out)
    print(f" Data Training Successful with Pipeline-> {out}")


if __name__ == "__main__":
    build_full_pipeline_with_smote(RAW_PATH)

