import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------
# Load cleaned dataset
# -------------------------------
input_path = "data/processed/cleaned_spam.csv"
df = pd.read_csv(input_path)

# DROP NaN text rows (VERY IMPORTANT)
df = df.dropna(subset=["clean_text"])

X = df["clean_text"]
y = df["label_num"]

print("Dataset loaded")
print("Total samples after NaN removal:", len(df))

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

print("TF-IDF transformation completed")
print("Feature shape:", X_tfidf.shape)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("Train-test split completed")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------------
# Save outputs
# -------------------------------
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("TF-IDF features and data splits saved successfully")
