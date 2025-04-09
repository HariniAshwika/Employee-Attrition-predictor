# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv("HR_Employee_Attrition.csv")

# Drop unnecessary columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode categorical features
le_dict = {}  # Dictionary to store LabelEncoders
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    le_dict[column] = le

# Define features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and encoders
with open("attrition_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)

# Predict on a sample
sample = X_test.iloc[0]
sample_reshaped = sample.values.reshape(1, -1)
pred = model.predict(sample_reshaped)
print("Predicted Attrition (1=Yes, 0=No):", pred[0])
