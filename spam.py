import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

# Load dataset
dataset_path = 'spam.csv'  
data = pd.read_csv(dataset_path, encoding='latin-1')

# Preview data
print("Dataset sample:")
print(data.head())

# Retain relevant columns and rename for clarity
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Encode target variable: 'spam' = 1, 'ham' = 0
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split dataset into training and test sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Convert text data to numerical using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test with some custom messages
custom_messages = ["Congratulations! You've won a $1000 gift card. Click here to claim.", 
                   "Hey, are we still meeting for dinner later?"]
custom_messages_tfidf = tfidf_vectorizer.transform(custom_messages)
custom_predictions = model.predict(custom_messages_tfidf)
custom_predictions_proba = model.predict_proba(custom_messages_tfidf)[:, 1]

print("\nCustom Messages Predictions:")
for i, msg in enumerate(custom_messages):
    print(f"Message: {msg}")
    print(f"Prediction: {'Spam' if custom_predictions[i] == 1 else 'Ham'} (Probability: {custom_predictions_proba[i]:.2f})")
