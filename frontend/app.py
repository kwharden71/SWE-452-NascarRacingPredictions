# =========================================================
# This file is the main entrance point for the streamlit UI
# =========================================================

import streamlit as st
import pandas as pd
import pickle
import os

# Load trained model
@st.cache_resource
def load_model():
    # Path relative to frontend folder
    model_path = os.path.join("..", "models", "lgbm_ranker.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Preprocessing
def preprocess(df):
    df = df.copy()
    df.fillna(0, inplace=True)

    drop_cols = ["id", "Driver", "Points", "Year", "Rank"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Match training feature order
    model_features = model.booster_.feature_name()
    X = X.reindex(columns=model_features, fill_value=0)

    return X

# Predict ranks
def predict(df):
    X = preprocess(df)
    df['pred_score'] = model.predict(X)
    df['predicted_rank'] = df['pred_score'].rank(
        ascending=False,
        method='min'
    ).astype(int)

    df = df.sort_values('predicted_rank')
    return df[['Driver', 'predicted_rank']]

# Streamlit UI
st.set_page_config(page_title="Nascar Race Predictions", layout="wide")
st.title(" Nascar Race Predictions")

# Automatically load CSV from data folder
csv_path = os.path.join("..", "data", "nascar_driver_statistics.csv")
new_data = pd.read_csv(csv_path)

st.subheader("Preview of dataset:")
st.dataframe(new_data.head())

if st.button("Predict Rankings"):
    predictions = predict(new_data)
    st.subheader("Predicted Rankings:")
    st.dataframe(predictions.reset_index(drop=True))
