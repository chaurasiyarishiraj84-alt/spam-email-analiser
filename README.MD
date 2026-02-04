this is the working link of my project =  https://huggingface.co/spaces/Rishi-raj123/spam-email-analyser 

# 📧 Spam Email Detection System

An end-to-end **Spam Email Detection System** built using **Natural Language Processing (NLP)** and **Machine Learning**, with a **Streamlit-based web interface** for real-time email classification.

This project demonstrates a complete ML pipeline — from data preprocessing and feature extraction to model training, prediction, logging, and deployment readiness.

--- 

## 🚀 Features

- Clean and preprocess raw email text
- TF-IDF feature extraction (unigrams + bigrams)
- Machine Learning model using Naive Bayes
- Persistent trained model and vectorizer
- Real-time spam/ham prediction
- Streamlit web application 
- Prediction logging for analysis
- Retraining-ready pipeline
- Industry-standard project structure

---

## 🧠 Tech Stack

- **Python**
- **scikit-learn**
- **pandas**
- **NumPy**
- **Streamlit**
- **Joblib**

---

## 📁 Project Structure

spam-email-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── retrain_model.py
│   ├── predict.py
│   └── model_evaluation.py
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── data/
│   ├── raw/
│   │   ├── spam_ham_dataset.csv
│   │   └── new_emails.csv
│   └── processed/
│       └── cleaned_spam.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── logs/
│   └── prediction_logs.csv
│
├── tests/
│   └── test_prediction.py
│
└── reports/
    └── metrics.txt

---

## 📊 Dataset

- Binary classification dataset with labels:
  - `1` → Spam
  - `0` → Ham
- Text data is cleaned and normalized before training

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash

git clone https://github.com/chaurasiyaishiraj84-alt/spam-email-analiser.git
cd spam-email-detection
2️⃣ Create virtual environment (optional but recommended)
python -m venv .venv

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Application
Streamlit Web App
streamlit run app.py
Then open the URL shown in the terminal and enter an email to classify.

CLI Prediction (optional)
python src/predict.py

🧪 Model Details
Vectorizer: TF-IDF (max_features=5000, ngram_range=(1,2))

Model: Multinomial Naive Bayes

Evaluation: Accuracy, Precision, Recall, F1-score (via model_evaluation.py)

🔁 Retraining Support
New emails can be logged and later used to retrain the model using:

python src/retrain_model.py
This allows continuous model improvement.

🌍 Deployment
The project is fully deployment-ready and can be hosted on:

Hugging Face Spaces

Streamlit Cloud

📌 Future Enhancements
Prediction confidence scores

User feedback-based retraining

REST API using FastAPI

Dockerization

Cloud deployment

👤 Author
Developed by [Rishi Raj Chaurasiya]
B.Tech in Artificial Intelligence & Machine Learning

📜 License
This project is open-source and available under the MIT License.

