import joblib
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -------------------------------
# Load trained model and vectorizer
# -------------------------------
model = joblib.load("models/spam_classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

print("Model and vectorizer loaded")

# -------------------------------
# Text cleaning function (same as preprocessing)
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"subject:", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    return " ".join(words)

# -------------------------------
# User input
# -------------------------------
email = input("\nEnter email text: ")

cleaned_email = clean_text(email)
email_vector = tfidf.transform([cleaned_email])

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(email_vector)[0]

if prediction == 1:
    print("\nPrediction: 🚨 SPAM")
else:
    print("\nPrediction: ✅ HAM (Not Spam)")
