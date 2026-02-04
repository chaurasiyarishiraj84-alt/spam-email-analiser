import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load training & testing data
# -------------------------------
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

print("Training and testing data loaded")

# -------------------------------
# Train model
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed")

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save trained model
# -------------------------------
joblib.dump(model, "models/spam_classifier.pkl")

print("Trained model saved successfully")
