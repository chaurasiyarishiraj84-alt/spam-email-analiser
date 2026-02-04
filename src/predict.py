import joblib
import pandas as pd
from datetime import datetime
import os
from data_preprocessing import clean_text

# -------------------------------
# Load trained model & vectorizer
# -------------------------------
model = joblib.load("models/spam_classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

print("Model and vectorizer loaded successfully")

# -------------------------------
# Take user input
# -------------------------------
email_text = input("\nEnter email text to classify:\n")

cleaned_email = clean_text(email_text)
email_tfidf = tfidf.transform([cleaned_email])

prediction = model.predict(email_tfidf)[0]
prediction_label = "SPAM" if prediction == 1 else "HAM"

print(f"\n📩 Prediction: {prediction_label}")

# -------------------------------
# LOGGING (FIXED)
# -------------------------------
os.makedirs("logs", exist_ok=True)
log_file = "logs/prediction_logs.csv"

log_row = pd.DataFrame({
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "email_text": [email_text],
    "prediction": [prediction_label]
})

# CASE 1 & 2: file missing OR empty
if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
    log_row.to_csv(log_file, index=False)
else:
    df_existing = pd.read_csv(log_file)
    df_updated = pd.concat([df_existing, log_row], ignore_index=True)
    df_updated.to_csv(log_file, index=False)

print("📝 Prediction logged successfully")
