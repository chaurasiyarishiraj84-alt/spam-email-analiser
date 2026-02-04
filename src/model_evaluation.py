import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -------------------------------
# Load model and test data
# -------------------------------
model = joblib.load("models/spam_classifier.pkl")
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

print("Model and test data loaded")

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Accuracy
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# -------------------------------
# Classification report
# -------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix Meaning:")
print("TN  FP")
print("FN  TP")
