import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load training and test datasets
train_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')

# Select relevant features and target variable
X_train_full = train_data.drop(['is_fraud', 'trans_date_trans_time', 'cc_num', 'first', 'last', 
                                'street', 'city', 'state', 'zip', 'lat', 'long', 'dob', 
                                'trans_num'], axis=1)
y_train_full = train_data['is_fraud']

X_test = test_data.drop(['Unnamed: 0','is_fraud', 'trans_date_trans_time', 'cc_num', 'first', 'last', 
                         'street', 'city', 'state', 'zip', 'lat', 'long', 'dob', 
                         'trans_num'], axis=1, errors='ignore')
print(test_data.columns)
y_test = test_data['is_fraud']

# Preprocessing pipeline for numerical and categorical features
numeric_features = ['amt', 'city_pop', 'merch_lat', 'merch_long']
categorical_features = ['category', 'gender']

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train_full)

# Split the data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train_preprocessed, y_train_full, 
                                                  test_size=0.3, random_state=42, stratify=y_train_full)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the logistic regression model
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Predictions and evaluation on validation data
y_val_pred = model.predict(X_val)
y_val_proba = model.predict_proba(X_val)[:, 1]

print("Classification Report on Validation Data:")
print(classification_report(y_val, y_val_pred))
print("ROC-AUC Score on Validation Data:", roc_auc_score(y_val, y_val_proba))
print("Confusion Matrix on Validation Data:\n", confusion_matrix(y_val, y_val_pred))

# Transform the test data using the same preprocessing pipeline
X_test_preprocessed = preprocessor.transform(X_test)

# Predict on the test data
y_test_pred = model.predict(X_test_preprocessed)
y_test_proba = model.predict_proba(X_test_preprocessed)[:, 1]

print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_test_pred))
print("ROC-AUC Score on Test Data:", roc_auc_score(y_test, y_test_proba))
print("Confusion Matrix on Test Data:\n", confusion_matrix(y_test, y_test_pred))

# Optionally, output the predictions alongside the original test data for review
test_data['is_fraud_predicted'] = y_test_pred
print("\nSample of Predictions on Test Data:")
print(test_data[['trans_date_trans_time', 'cc_num', 'merchant', 'amt', 'is_fraud', 'is_fraud_predicted']].head(10))
