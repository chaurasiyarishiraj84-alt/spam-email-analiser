import pandas as pd
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load raw dataset
input_path = "data/raw/spam_ham_dataset.csv"
df = pd.read_csv(input_path, encoding="latin-1")


print("Dataset loaded successfully")
print("Initial shape:", df.shape)

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"subject:", "", text)         # remove 'subject'
    text = re.sub(r"\d+", "", text)              # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = text.strip()                          # remove extra spaces
    
    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    return " ".join(words)

# -------------------------------
# Apply cleaning
# -------------------------------
df["clean_text"] = df["text"].apply(clean_text)

# -------------------------------
# Keep required columns
# -------------------------------
clean_df = df[["clean_text", "label_num"]]

print("Preprocessing completed")
print("Final shape:", clean_df.shape)

# Save processed data
output_path = "data/processed/cleaned_spam.csv"
clean_df.to_csv(output_path, index=False)
