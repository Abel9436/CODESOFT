# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Function to parse each line in the dataset
def parse_line(line, has_genre=True):
    parts = line.strip().split(" ::: ")
    if has_genre:
        return {"Title (Year)": parts[1], "Genre": parts[2], "Plot": parts[3]}
    else:
        return {"Title (Year)": parts[1], "Plot": parts[2]}

# Function to load data from a file
def load_data(file_path, has_genre=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(parse_line(line, has_genre=has_genre))
    return pd.DataFrame(data)

# Load the training data
train_data = load_data("train_data.txt", has_genre=True)

# Initialize the TF-IDF Vectorizer and transform data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(train_data['Plot'])
y = train_data['Genre']

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test set
predictions = log_reg.predict(X_test)

# Evaluate the model on the test set
print("Test Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Optional: Predict on new data (test set) if provided
test_data = load_data("test_data.txt", has_genre=False)  # Load test data without genre labels
test_tfidf = vectorizer.transform(test_data['Plot'])  # Transform using the same vectorizer
test_predictions = log_reg.predict(test_tfidf)

# Display test predictions
test_data['Predicted Genre'] = test_predictions
print("\nTest Data Predictions:\n", test_data[['Title (Year)', 'Predicted Genre']])
