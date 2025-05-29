import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('dataset/records.csv')

# Separate features and target
X = df.drop(['time', 'result'], axis=1)  # Remove time and result columns
y = df['result']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100

# Print results
print("\nModel Performance Analysis:")
print("-" * 30)
print(f"Accuracy: {accuracy:.2f}%")
print("\nDetailed Classification Report:")
print("-" * 30)
print(classification_report(y_test, y_pred))

# Calculate and print total records and distribution
total_records = len(df)
healthy_records = len(df[df['result'] == 0])
disease_records = len(df[df['result'] == 1])

print("\nDataset Statistics:")
print("-" * 30)
print(f"Total Records: {total_records}")
print(f"Healthy Patients: {healthy_records} ({healthy_records/total_records*100:.1f}%)")
print(f"Patients with Kidney Disease: {disease_records} ({disease_records/total_records*100:.1f}%)") 