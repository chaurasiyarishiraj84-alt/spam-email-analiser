import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from data_preprocessing import clean_text


# ===============================
# Load original cleaned dataset
# ===============================
OLD_DATA_PATH = "data/processed/cleaned_spam.csv"
NEW_DATA_PATH = "data/raw/new_emails.csv"

df_old = pd.read_csv(OLD_DATA_PATH)

# Validate old dataset
required_cols = {"clean_text", "label_num"}
if not required_cols.issubset(df_old.columns):
    raise ValueError("cleaned_spam.csv must contain 'clean_text' and 'label_num'")


# ===============================
# Load new user-added emails
# ===============================
if os.path.exists(NEW_DATA_PATH):
    df_new = pd.read_csv(NEW_DATA_PATH)

    # Normalize column names
    df_new.columns = [col.lower().strip() for col in df_new.columns]

    if not {"text", "label_num"}.issubset(df_new.columns):
        raise ValueError("new_emails.csv must contain 'text' and 'label_num' columns")

    # Clean new email text
    df_new["clean_text"] = df_new["text"].apply(clean_text)
    df_new = df_new.dropna(subset=["clean_text"])
    df_new = df_new[df_new["clean_text"].str.strip() != ""]

    # Keep only required columns
    df_new = df_new[["clean_text", "label_num"]]

else:
    print("No new emails found. Using only existing dataset.")
    df_new = pd.DataFrame(columns=["clean_text", "label_num"])


# ===============================
# Combine old + new datasets
# ===============================
df_combined = pd.concat([df_old, df_new], ignore_index=True)

# Remove duplicate emails
df_combined = df_combined.drop_duplicates(subset=["clean_text"], keep="first")

# Remove NaN or empty text
df_combined = df_combined.dropna(subset=["clean_text"])
df_combined = df_combined[df_combined["clean_text"].str.strip() != ""]


# ===============================
# FIX LABEL DATATYPE (CRITICAL)
# ===============================
df_combined["label_num"] = pd.to_numeric(
    df_combined["label_num"], errors="coerce"
)

df_combined = df_combined.dropna(subset=["label_num"])
df_combined["label_num"] = df_combined["label_num"].astype(int)


print(f"Total samples after cleaning: {len(df_combined)}")
print("Label values:", df_combined["label_num"].unique())


# ===============================
# Feature Extraction (TF-IDF)
# ===============================
X = df_combined["clean_text"]
y = df_combined["label_num"]

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)


# ===============================
# Train Naive Bayes Model
# ===============================
model = MultinomialNB()
model.fit(X_tfidf, y)


# ===============================
# Save Model & Vectorizer
# ===============================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("✅ Model retrained successfully with new emails!")


# ===============================
# Clear temporary new emails file
# ===============================
pd.DataFrame(columns=["text", "label_num"]).to_csv(
    NEW_DATA_PATH, index=False
)

print("🧹 Cleared new_emails.csv to avoid duplicate retraining.")
