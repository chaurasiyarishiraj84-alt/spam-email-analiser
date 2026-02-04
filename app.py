import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

from src.data_preprocessing import clean_text


# Load model & vectorizer

@st.cache_resource
def load_model():
    model = joblib.load("models/spam_classifier.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

# UI Layout

st.set_page_config(page_title="Spam Email Detector", layout="centered")
st.title("📧 Spam Email Detection System")
st.write("Enter an email below to check whether it is **Spam** or **Ham**.")

email_text = st.text_area("Email content", height=200)


# Prediction

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:
        cleaned = clean_text(email_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "SPAM 🚫" if prediction == 1 else "HAM ✅"

        st.subheader("Result")
        st.success(label) if prediction == 0 else st.error(label)

        
        # Logging
       
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/prediction_logs.csv"

        log_row = pd.DataFrame({
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "email_text": [email_text],
            "prediction": ["SPAM" if prediction == 1 else "HAM"]
        })

        if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
            log_row.to_csv(log_file, index=False)
        else:
            df_existing = pd.read_csv(log_file)
            df_updated = pd.concat([df_existing, log_row], ignore_index=True)
            df_updated.to_csv(log_file, index=False)

        st.info("Prediction logged successfully.")
