import pandas as pd
import os

new_emails_file = "data/raw/new_emails.csv"

# Get input
email_text = input("Enter the email text: ")
label = input("Enter label (1=spam, 0=ham): ")

# Validate label
if label not in ["0", "1"]:
    print("Invalid label! Enter 0 or 1.")
    exit()

new_data = pd.DataFrame([[email_text, int(label)]], columns=["text","label_num"])

# Append safely
if os.path.exists(new_emails_file):
    df_existing = pd.read_csv(new_emails_file)
    
    # Optional: remove duplicates before appending
    df_existing = df_existing.drop_duplicates(subset=['text'], keep='first')
    
    df_updated = pd.concat([df_existing, new_data], ignore_index=True)
else:
    df_updated = new_data

# Save CSV safely
df_updated.to_csv(new_emails_file, index=False)
print("Email added successfully!")
